from utils import size
from typing import List, Tuple, Optional
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy


class BatchedMatmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [b, M, K] * [b, K, N] = [b, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        assert size(self.input1_shape[:-2]) == size(self.input2_shape[:-2])
        self.bs = size(self.input1_shape[:-2])
        self.M = self.input1_shape[-2]
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        self.output_shape = self.input1_shape[:-2] + [self.M, self.N]
        output = Tensor(self.output_shape, self.data_type)
        return output

    def roofline_model(self, pcb_module: Device):
        matmul = Matmul(self.data_type)
        _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
        matmul_latency = matmul.roofline_model(pcb_module)
        self.roofline_latency = matmul_latency * self.bs
        return self.roofline_latency

    # def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
    #     matmul = Matmul(self.data_type)
    #     _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
    #     matmul_latency = (
    #         matmul.compile_and_simulate(pcb_module, compile_mode)
    #         # - pcb_module.io_module.latency * 2
    #     )
    #     self.latency = matmul_latency * self.bs  # + pcb_module.io_module.latency * 2
    #     return self.latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        # 策略1：将批量矩阵乘法分解为多个独立的矩阵乘法
        matmul = Matmul(self.data_type)
        _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
        matmul_latency1 = (
            matmul.compile_and_simulate(pcb_module, compile_mode) * self.bs
        )
        
        print(f"#1 Matmul latency per batch: {matmul_latency1/self.bs*1e3:.4f} ms, Total latency for {self.bs} batches: {matmul_latency1*1e3:.4f} ms")
        
        # 策略2：将批量矩阵乘法作为一个整体进行处理 
        matmul = Matmul(self.data_type)
        _ = matmul(
            Tensor([self.M, self.K * self.bs]), Tensor([self.K * self.bs, self.N])
        )
        # todo:latency2的后半部分的数据读取的延迟需要修改为HBM架构
        matmul_latency2 = (
            matmul.compile_and_simulate(pcb_module, compile_mode)
            + (self.bs - 1)
            * self.M
            * self.N
            * self.data_type.word_size
            / pcb_module.io_module.bandwidth
        )
        
        print(f"#2 Matmul latency treating batch as whole: {matmul_latency2*1e3:.4f} ms")
        
        # 选择两种策略中latency较小的作为最终结果
        self.latency = min(matmul_latency1, matmul_latency2)
        return self.latency

    def run_on_gpu(
        self,
    ):
        input1 = torch.randn(self.bs, self.M, self.K, dtype=torch.float16).cuda()
        input2 = torch.randn(self.bs, self.K, self.N, dtype=torch.float16).cuda()
        latencies = []
        # warmup
        for _ in range(3):
            _ = torch.bmm(input1, input2)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = torch.bmm(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        latencies = []
        for _ in range(50):
            a = torch.randn(1, 1, 1, device="cuda")
            b = torch.randn(1, 1, 1, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.bmm(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        # print(latencies)
        return avg_overhead


class Matmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None
        self.best_stacked_mapping = None

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K + self.K * self.N + self.M * self.N
        # print(f'{self.M}, {self.N}, {self.K}')
        return output

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = max(
            self.flop_count / pcb_module.compute_module.total_systolic_array_flops,
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
        )
        return self.roofline_latency

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    
    
    class Stacked_Mapping:
        def __init__(
            self,
            HBM_tile_M: int,
            HBM_tile_N: int,
            HBM_tile_K: int,
            HBM_double_buffering: bool,
            core_tile_M: int,
            core_tile_N: int,
            core_tile_K: int,
            # l2_loop_order: str,
            core_loop_order: str,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            dataflow: str = "os",
            # 新增：HBM 批次调度策略
            hbm_schedule: str = "diagonal-channel",  # 暂时先考虑对角线调度，可选: row-major | col-major | diagonal | diagonal-channel | custom
            custom_batches: Optional["List[List[Tuple[int,int,int]]]"] = None,  # 若 hbm_schedule == custom，则使用该批次定义
        ):
            self.HBM_tile_M = HBM_tile_M
            self.HBM_tile_N = HBM_tile_N
            self.HBM_tile_K = HBM_tile_K
            self.HBM_double_buffering = HBM_double_buffering
            self.core_tile_M = core_tile_M
            self.core_tile_N = core_tile_N
            self.core_tile_K = core_tile_K
            # self.l2_loop_order = l2_loop_order
            self.core_loop_order = core_loop_order
            self.l0_M_tiling_factor = l0_M_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor
            self.dataflow = dataflow
            self.hbm_schedule = hbm_schedule
            self.custom_batches = custom_batches
    
    @staticmethod
    def find_permutations(n): #这个函数仅用于L0级的脉动阵列优化处理
        permutations = set() #set集合，用于存储所有可能的排列组合，集合不允许重复，元素是无序的

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations) #将集合转换为列表，返回所有可能的排列组合

    @staticmethod
    def generate_hbm_batches(
        M_tiles: int,
        N_tiles: int,
        K_tiles: int,
        channel_count: int,
        schedule: str,
        custom_batches: Optional["List[List[Tuple[int,int,int]]]"] = None,
    ) -> "List[List[Tuple[int,int,int]]]":
        """生成分批执行的 HBM tile 列表。
        返回值为批次列表；每个批次是 (m_idx, n_idx, k_idx) 元组的列表；
        同一批次内的 tile 视为并行执行，批次时长取该批次内最长 tile 时长。
        预置策略：
        - row-major: 先按 k，从小到大；k 固定时按 m 行、n 列扫描，将每批最多 channel_count 个 tile 打包。
        - col-major: 与上类似，但 n 优先。
        - diagonal: 对 (m, n) 网格按反对角线 (m+n = 常数) 切片，每片最多 channel_count 个。
        - diagonal-channel: 针对 M_tiles == N_tiles 的“对角移位”调度；
          第 b 批为 core c 分配 (m=c, n=(c+b)%N_tiles, k=0)。若 K_tiles>1，则对每个 k 顺序重复该模式。
        - custom: 使用 custom_batches 直接指定批次。

        - todo:这一块的函数要仔细检查
        """
        batches: List[List[Tuple[int,int,int]]] = []

        if schedule == "custom":
            return custom_batches or []

        def chunk(items: List[Tuple[int,int,int]], n: int) -> List[List[Tuple[int,int,int]]]:
            return [items[i:i+n] for i in range(0, len(items), n)]

        if schedule == "row-major":
            for k in range(K_tiles):
                flat: List[Tuple[int,int,int]] = []
                for m in range(M_tiles):
                    for n in range(N_tiles):
                        flat.append((m, n, k))
                batches.extend(chunk(flat, channel_count))
            return batches

        if schedule == "col-major":
            for k in range(K_tiles):
                flat: List[Tuple[int,int,int]] = []
                for n in range(N_tiles):
                    for m in range(M_tiles):
                        flat.append((m, n, k))
                batches.extend(chunk(flat, channel_count))
            return batches

        if schedule == "diagonal":
            for k in range(K_tiles):
                # 反对角线：s = m + n，从 0 到 (M_tiles+N_tiles-2)
                for s in range(M_tiles + N_tiles - 1):
                    diag: List[Tuple[int,int,int]] = []
                    m_start = max(0, s - (N_tiles - 1))
                    m_end = min(M_tiles - 1, s)
                    for m in range(m_start, m_end + 1):
                        n = s - m
                        diag.append((m, n, k))
                    batches.extend(chunk(diag, channel_count))
            return batches

        if schedule == "diagonal-channel":
            # 针对“对角运算”需求；要求 M_tiles 与 N_tiles 至少覆盖 channel_count
            # k 维度顺序执行；每个 k 上进行 B = max(M_tiles, N_tiles) 个批次
            # C = min(M_tiles, N_tiles)
            # if C == 0:
            #     return []
            # B = max(M_tiles, N_tiles)
            for k in range(K_tiles):
                for b in range(channel_count): # b 表示第 b 批，即相对于对角线的偏移量
                    batch: List[Tuple[int,int,int]] = []
                    # 为每个 core 分配一个 (m, n)
                    for c in range(channel_count): # c 表示 core 的编号
                        m = c % M_tiles
                        n = (c + b) % N_tiles
                        batch.append((m, n, k))
                    if batch:
                        batches.append(batch)
            return batches

        # 默认退化为 row-major
        for k in range(K_tiles):
            flat: List[Tuple[int,int,int]] = []
            for m in range(M_tiles):
                for n in range(N_tiles):
                    flat.append((m, n, k))
            batches.extend(chunk(flat, channel_count))
        return batches

    def compile_and_simulate(  #核心功能是遍历生成mapping
        self,
        pcb_module: Device,
        compile_mode: str = "3D_stacked",
    ):
        print(f"\n{'='*70}")
        print(f"Starting compile_and_simulate - Mode: {compile_mode}")
        print(f"{'='*70}")
        print(f"Computational Graph: M={self.computational_graph.M}, N={self.computational_graph.N}, K={self.computational_graph.K}")
        print(f"Device Configuration:")
        print(f"  - device name: {pcb_module}")
        print(f"  - Core count: {pcb_module.compute_module.core_count}")
        print(f"  - Memory Channels: {pcb_module.memory_module.channel_count}")
        print(f"  - Memory Capacity per Channel: {pcb_module.memory_module.memory_capacity / pcb_module.memory_module.channel_count / 1024 / 1024 / 1024:.2f} GB")
        print(f"  - Core SRAM: {pcb_module.compute_module.core.SRAM_size / 1024:.1f} KB")
        print(f"  - Systolic Array: {pcb_module.compute_module.core.systolic_array.array_height}x{pcb_module.compute_module.core.systolic_array.array_width}")
        print(f"  - Channel Bandwidth: {pcb_module.compute_module.channel_bandwidth_per_cycle} B/cycle")
        print(f"{'='*70}\n")
        
        min_cycle_count = 2**63 - 1
        best_mapping = None
        best_stacked_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        # if (M == 1 or N == 1) and (
        #     compile_mode == "heuristic-GPU"
        #     or compile_mode == "heuristic-our-throughput"
        # ):
        #     working_set_size = M * K + N * K + M * N
        #     total_io_count = working_set_size * self.data_type.word_size
        #     io_latency = total_io_count / pcb_module.io_module.bandwidth
        #     total_flop_count = 2 * M * N * K
        #     compute_latency = (
        #         total_flop_count
        #         / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
        #         / pcb_module.compute_module.core_count
        #         / pcb_module.compute_module.clock_freq
        #     )
        #     self.latency = max(
        #         compute_latency, io_latency
        #     )  # + pcb_module.io_module.latency * 2
        #     return self.latency
        if compile_mode == "3D_stacked":
            HBM_tile_M_log2 = ceil(log2(self.computational_graph.M / pcb_module.memory_module.channel_count)) #每个HBM tile的M维度大小为按channel等分后+1的二的次方，最后的多余部分采用remain进行单独计算
            HBM_tile_N_log2 = ceil(log2(self.computational_graph.N / pcb_module.memory_module.channel_count))
            HBM_tile_K_log2 = ceil(log2(self.computational_graph.K))#K维度暂时不进行切分
            HBM_TILE_M = 2 ** HBM_tile_M_log2
            HBM_TILE_N = 2 ** HBM_tile_N_log2
            HBM_TILE_K = 2 ** HBM_tile_K_log2

            working_set_size = (
                HBM_TILE_N * HBM_TILE_K
                + HBM_TILE_M * HBM_TILE_K
                + HBM_TILE_M * HBM_TILE_N
            )
            print(f"HBM Tile Size: M={HBM_TILE_M}, N={HBM_TILE_N}, K={HBM_TILE_K}")
            print(f"HBM Working Set Size: {working_set_size * self.data_type.word_size / 1024:.1f} KB per channel")
            
            assert working_set_size <= pcb_module.memory_module.memory_capacity / pcb_module.memory_module.channel_count // self.data_type.word_size, "HBM tile size exceeds memory capacity"
            
            if (
                working_set_size
                <= (pcb_module.memory_module.memory_capacity / pcb_module.memory_module.channel_count)
                // self.data_type.word_size
                // 2
            ):
                HBM_double_buffering = True
            else:
                HBM_double_buffering = False
            
            print(f"HBM Double Buffering: {HBM_double_buffering}")
            print(f"\nSearching optimal mapping configurations...")
            
            total_configs = 0
            tested_configs = 0
            for core_tile_M_log2 in range(5, HBM_tile_M_log2 + 1):
                core_tile_M = 2**core_tile_M_log2
                for core_tile_N_log2 in range(5, HBM_tile_N_log2 + 1):
                    core_tile_N = 2**core_tile_N_log2
                    for core_tile_K_log2 in range(5, HBM_tile_K_log2 + 1):
                        core_tile_K = 2**core_tile_K_log2
                        if (
                            core_tile_M * core_tile_N
                            + core_tile_N * core_tile_K
                            + core_tile_M * core_tile_K
                            > pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                        ):
                            continue
                        for core_loop_order in [
                            "mkn",
                            "mnk",
                            "nkm",
                            "nmk",
                            "knm",
                            "kmn",
                        ]:
                            for (
                                l0_M_tiling_factor,
                                l0_N_tiling_factor,
                                l0_K_tiling_factor,
                            ) in self.find_permutations(
                                pcb_module.compute_module.core.systolic_array_count
                            ):
                                stacked_mapping = self.Stacked_Mapping(
                                    HBM_TILE_M,
                                    HBM_TILE_N,
                                    HBM_TILE_K,
                                    HBM_double_buffering,
                                    core_tile_M,
                                    core_tile_N,
                                    core_tile_K,
                                    core_loop_order,
                                    l0_M_tiling_factor,
                                    l0_N_tiling_factor,
                                    l0_K_tiling_factor,
                                )
                                tested_configs += 1
                                if tested_configs % 50 == 0 or tested_configs == 1:
                                    print(f"  Progress: tested {tested_configs} configs | Current best: {min_cycle_count / pcb_module.compute_module.clock_freq * 1000:.3f} ms")
                                    print(f"    → Core tile: M={core_tile_M}, N={core_tile_N}, K={core_tile_K}, loop={core_loop_order}")
                                
                                cycle_count = self.stacked_simulate(
                                    self.computational_graph,
                                    stacked_mapping,
                                    pcb_module,
                                )
                                if cycle_count < min_cycle_count:
                                    min_cycle_count = cycle_count
                                    best_stacked_mapping = stacked_mapping
                                    print(f"  ✓ New best found! Cycle count: {cycle_count}, Latency: {cycle_count / pcb_module.compute_module.clock_freq * 1000:.3f} ms")
        else:
            raise ValueError(f"compile_mode {compile_mode} not supported")
        
        print(f"\n{'='*70}")
        print(f"Optimization Complete!")
        print(f"  - Total configurations tested: {tested_configs}")
        print(f"  - Best cycle count: {min_cycle_count}")
        print(f"  - Best latency: {min_cycle_count / pcb_module.compute_module.clock_freq * 1000:.3f} ms")
        if best_stacked_mapping:
            print(f"  - Best HBM tile: M={best_stacked_mapping.HBM_tile_M}, N={best_stacked_mapping.HBM_tile_N}, K={best_stacked_mapping.HBM_tile_K}")
            print(f"  - Best core tile: M={best_stacked_mapping.core_tile_M}, N={best_stacked_mapping.core_tile_N}, K={best_stacked_mapping.core_tile_K}")
            print(f"  - Loop order: {best_stacked_mapping.core_loop_order}")
        print(f"{'='*70}\n")
        
        self.best_mapping = best_mapping
        self.best_stacked_mapping = best_stacked_mapping
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency

    
    def stacked_simulate( #根据mapping进行任务分割
        self,
        computational_graph:ComputationalGraph,
        stacked_mapping:Stacked_Mapping,
        pcb_module:Device,
    ) -> int:
        if self.look_up_table is None:
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
        # print(self.look_up_table)
        # print(self.look_up_table.loc[(32, 16, 256, 16, 16, 'os'), "cycle_count"
        #                              ].item())
        # print('sdfsdfsdfsd')
        # exit()
        
        #总矩阵大小
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        HBM_tile_M = stacked_mapping.HBM_tile_M
        HBM_tile_N = stacked_mapping.HBM_tile_N
        HBM_tile_K = stacked_mapping.HBM_tile_K

        if stacked_mapping.HBM_double_buffering:
            assert (
                HBM_tile_M * HBM_tile_N + HBM_tile_N * HBM_tile_K + HBM_tile_M * HBM_tile_K
                <= (pcb_module.memory_module.memory_capacity / pcb_module.memory_module.channel_count) // self.data_type.word_size // 2
            )
        else:
            assert (
                HBM_tile_M * HBM_tile_N + HBM_tile_N * HBM_tile_K + HBM_tile_M * HBM_tile_K
                <= (pcb_module.memory_module.memory_capacity / pcb_module.memory_module.channel_count) // self.data_type.word_size
            )

        
        M_HBM_t = M // HBM_tile_M
        N_HBM_t = N // HBM_tile_N
        K_HBM_t = K // HBM_tile_K 
        # //这里的K维度切分可能需要修改,因为选择的HBM tile的K维度基本大于实际的K维度，而实际不打算对K维度进行切分
        # 经过检查应该不需要修改，可以在K_remain里被正确的初始化
        M_remain = M % HBM_tile_M
        N_remain = N % HBM_tile_N
        K_remain = K % HBM_tile_K

        hbm_tiles = np.empty( # 创建一个三维数组，将每一个子矩阵乘作为一个对象，类型是HBMTileSimulator
            [ceil(M / HBM_tile_M), ceil(N / HBM_tile_N), ceil(K / HBM_tile_K)],
            dtype=self.HBMTileSimulator,
        )
        
        if M_HBM_t * N_HBM_t * K_HBM_t != 0:
            hbm_tiles[:M_HBM_t, :N_HBM_t, :K_HBM_t] = self.HBMTileSimulator(
                HBM_tile_M,
                HBM_tile_N,
                HBM_tile_K,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain != 0:
            hbm_tiles[-1, :N_HBM_t, :K_HBM_t] = self.HBMTileSimulator(
                M_remain,
                HBM_tile_N,
                HBM_tile_K,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0:
            hbm_tiles[:M_HBM_t, -1, :K_HBM_t] = self.HBMTileSimulator(
                HBM_tile_M,
                N_remain,
                HBM_tile_K,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0:
            hbm_tiles[:M_HBM_t, :N_HBM_t, -1] = self.HBMTileSimulator(
                HBM_tile_M,
                HBM_tile_N,
                K_remain,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain != 0:
            hbm_tiles[-1, -1, :K_HBM_t] = self.HBMTileSimulator(
                M_remain,
                N_remain,
                HBM_tile_K,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain != 0:    
            hbm_tiles[-1, :N_HBM_t, -1] = self.HBMTileSimulator(
                M_remain,
                HBM_tile_N,
                K_remain,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0:
            hbm_tiles[:M_HBM_t, -1, -1] = self.HBMTileSimulator(
                HBM_tile_M,
                N_remain,
                K_remain,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain != 0:
            hbm_tiles[-1, -1, -1] = self.HBMTileSimulator(
                M_remain,
                N_remain,
                K_remain,
                data_type,
                stacked_mapping,
                pcb_module,
                self.look_up_table,
            )

        total_cycle_count = 0

        
        # *生成批次调度,采用channel_count个通道并行处理
        channel_count = pcb_module.memory_module.channel_count
        batches = self.generate_hbm_batches(
            M_tiles=ceil(M / HBM_tile_M),
            N_tiles=ceil(N / HBM_tile_N),
            K_tiles=ceil(K / HBM_tile_K),
            channel_count=channel_count,
            schedule=getattr(stacked_mapping, "hbm_schedule", "row-major"),
            custom_batches=getattr(stacked_mapping, "custom_batches", None),
        )

        for batch in batches:
            # 同一批次内并行执行，取最长 tile 时间
            if not batch:
                continue
            batch_cycles = 0
            for (m_idx, n_idx, k_idx) in batch: #遍历整个batch（并行同一批次内的HBM tile）
                # 越界保护（最后一片包含边界剩余块）
                if m_idx >= hbm_tiles.shape[0] or n_idx >= hbm_tiles.shape[1] or k_idx >= hbm_tiles.shape[2]:
                    continue
                tile = hbm_tiles[m_idx, n_idx, k_idx]
                batch_cycles = max(batch_cycles, tile.compute_cycle_count)  # 取最长 HBM tile 执行时间
            total_cycle_count += batch_cycles

        return total_cycle_count

    
    class HBMTileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            Stacked_Mapping: "Matmul.Stacked_Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'HBM tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type
            self.Stacked_Mapping = Stacked_Mapping
            self.pcb_module = pcb_module
            self.look_up_table = look_up_table
            # 在初始化时计算执行周期数
            self.compute_cycle_count = self.simulate_hbm_tile_compute_cycle_count(
                M, N, K, data_type, Stacked_Mapping, pcb_module, look_up_table
            )

        def simulate_hbm_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            Stacked_Mapping: "Matmul.Stacked_Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            core_tile_M = Stacked_Mapping.core_tile_M
            core_tile_N = Stacked_Mapping.core_tile_N
            core_tile_K = Stacked_Mapping.core_tile_K

            M_core_t = M // core_tile_M
            N_core_t = N // core_tile_N
            K_core_t = K // core_tile_K
            M_remain = M % core_tile_M
            N_remain = N % core_tile_N
            K_remain = K % core_tile_K

            # 创建一个三维数组，将HBM tile中的每一个子矩阵乘作为一个对象，类型是CoreTileSimulator
            core_tiles = np.empty(
                [ceil(M / core_tile_M), ceil(N / core_tile_N), ceil(K / core_tile_K)],
                dtype=Matmul.CoreTileSimulator,
            )
            # 对每个HBM tile中的每个core tile进行初始化
            if M_core_t * N_core_t * K_core_t != 0:
                core_tiles[:M_core_t, :N_core_t, :K_core_t] = Matmul.CoreTileSimulator(
                    core_tile_M,
                    core_tile_N,
                    core_tile_K,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            if M_remain != 0:
                core_tiles[-1, :N_core_t, :K_core_t] = Matmul.CoreTileSimulator(
                    M_remain,
                    core_tile_N,
                    core_tile_K,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            if N_remain != 0:
                core_tiles[:M_core_t, -1, :K_core_t] = Matmul.CoreTileSimulator(
                    core_tile_M,
                    N_remain,
                    core_tile_K,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            if K_remain != 0:
                core_tiles[:M_core_t, :N_core_t, -1] = Matmul.CoreTileSimulator(
                    core_tile_M,
                    core_tile_N,
                    K_remain,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            if M_remain * N_remain != 0:
                core_tiles[-1, -1, :K_core_t] = Matmul.CoreTileSimulator(
                    M_remain,
                    N_remain,
                    core_tile_K,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            if M_remain * K_remain != 0:
                core_tiles[-1, :N_core_t, -1] = Matmul.CoreTileSimulator(
                    M_remain,
                    core_tile_N,
                    K_remain,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            if N_remain * K_remain != 0:
                core_tiles[:M_core_t, -1, -1] = Matmul.CoreTileSimulator(
                    core_tile_M,
                    N_remain,
                    K_remain,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            if M_remain * N_remain * K_remain != 0:
                core_tiles[-1, -1, -1] = Matmul.CoreTileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    data_type,
                    Stacked_Mapping,
                    pcb_module,
                    look_up_table,
                )
            
            # //从这里开始往下应该是core_tile的运算部分，不太确定需不需要修改，先照L2 Tile的方式写着
            # 计算每个 core tile 的 M_K, K_N, M_N 矩阵大小
            M_K_tile_size = np.zeros(
                [ceil(M / core_tile_M), ceil(K / core_tile_K)], dtype=int
            )
            M_K_tile_size[:M_core_t, :K_core_t] = core_tile_M * core_tile_K
            if M_remain > 0:
                M_K_tile_size[-1, :K_core_t] = M_remain * core_tile_K
            if K_remain > 0:
                M_K_tile_size[:M_core_t, -1] = core_tile_M * K_remain
            if M_remain > 0 and K_remain > 0:
                M_K_tile_size[-1, -1] = M_remain * K_remain

            K_N_tile_size = np.zeros(
                [ceil(K / core_tile_K), ceil(N / core_tile_N)], dtype=int
            )
            K_N_tile_size[:K_core_t, :N_core_t] = core_tile_K * core_tile_N
            if K_remain > 0:
                K_N_tile_size[-1, :N_core_t] = K_remain * core_tile_N
            if N_remain > 0:
                K_N_tile_size[:K_core_t, -1] = core_tile_K * N_remain
            if K_remain > 0 and N_remain > 0:
                K_N_tile_size[-1, -1] = K_remain * N_remain

            M_N_tile_size = np.zeros(
                [ceil(M / core_tile_M), ceil(N / core_tile_N)], dtype=int
            )
            M_N_tile_size[:M_core_t, :N_core_t] = core_tile_M * core_tile_N
            if M_remain > 0:
                M_N_tile_size[-1, :N_core_t] = M_remain * core_tile_N
            if N_remain > 0:
                M_N_tile_size[:M_core_t, -1] = core_tile_M * N_remain
            if M_remain > 0 and N_remain > 0:
                M_N_tile_size[-1, -1] = M_remain * N_remain
            # todo:从这里往下进行了大幅度的修改，需要重点检查

            # 支持可配置的 m channel - n core 架构
            # 从 device 读取配置：假设 channel_count 和 core_count 定义在 compute_module 中
            # 如果 compute_module 中有 channel_count，则 cores_per_channel = core_count / channel_count
            # channel_count在memory_module中
            
            total_core_count = pcb_module.compute_module.core_count
            channel_count = pcb_module.memory_module.channel_count             
            cores_per_channel = total_core_count // channel_count if channel_count > 0 else total_core_count
            
            total_cycle_count = 0
            
            # 根据每个 channel 的 core 数量决定并行度
            if cores_per_channel == 1:
                # *单核模式：完全串行，保留数据重用
                # 读取第一个 core tile 的数据
                total_cycle_count += ceil(
                    (M_K_tile_size[0, 0] + K_N_tile_size[0, 0])
                    * data_type.word_size
                    / pcb_module.compute_module.channel_bandwidth_per_cycle
                )
                
                previous_m = 0
                previous_n = 0
                previous_k = 0
                
                for m, n, k in Matmul.generate_tile_loops(
                    ceil(M / core_tile_M),
                    ceil(N / core_tile_N),
                    ceil(K / core_tile_K),
                    Stacked_Mapping.core_loop_order,
                ):
                    # 跳过第一个 core tile的执行阶段
                    if m == 0 and n == 0 and k == 0:
                        continue
                    
                    core_tile = core_tiles[m, n, k]
                    previous_core_tile = core_tiles[previous_m, previous_n, previous_k]
                    
                    current_tile_read_cycle_count = 0
                    if m != previous_m or k != previous_k:
                        current_tile_read_cycle_count += ceil(
                            M_K_tile_size[m, k]
                            * data_type.word_size
                            / pcb_module.compute_module.channel_bandwidth_per_cycle
                        )
                    if k != previous_k or n != previous_n:
                        current_tile_read_cycle_count += ceil(
                            K_N_tile_size[k, n]
                            * data_type.word_size
                            / pcb_module.compute_module.channel_bandwidth_per_cycle
                        )
                    if k > 0 and (m != previous_m or n != previous_n):
                        current_tile_read_cycle_count += ceil(
                            M_N_tile_size[m, n]
                            * data_type.word_size
                            / pcb_module.compute_module.channel_bandwidth_per_cycle
                        )
                    
                    previous_tile_compute_cycle_count = previous_core_tile.compute_cycle_count
                    if previous_k > 0:
                        previous_tile_compute_cycle_count += ceil(
                            previous_core_tile.M
                            * previous_core_tile.N
                            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                        )
                    
                    if m == previous_m and n == previous_n:
                        previous_tile_write_cycle_count = 0
                    else:
                        previous_tile_write_cycle_count = ceil(
                            M_N_tile_size[previous_m, previous_n]
                            * data_type.word_size
                            / pcb_module.compute_module.channel_bandwidth_per_cycle #更换计算目标矩阵时写回数据到HBM中
                        )
                    
                    total_cycle_count += (
                        max(
                            current_tile_read_cycle_count,
                            previous_tile_compute_cycle_count,
                        )
                        + previous_tile_write_cycle_count
                    )
                    
                    previous_m = m
                    previous_n = n
                    previous_k = k
                
                last_core_tile = core_tiles[previous_m, previous_n, previous_k]
                last_tile_compute_cycle_count = last_core_tile.compute_cycle_count
                if previous_k > 0:
                    last_tile_compute_cycle_count += ceil(
                        last_core_tile.M
                        * last_core_tile.N
                        / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                    )
                last_tile_write_cycle_count = ceil(
                    M_N_tile_size[previous_m, previous_n]
                    * data_type.word_size
                    / pcb_module.compute_module.channel_bandwidth_per_cycle
                )
                total_cycle_count += last_tile_compute_cycle_count + last_tile_write_cycle_count
            
            else:
                # *多核模式：批量并行执行，保留数据重用，基本复用了L2 Tile中对L1tile的遍历逻辑
                previous_batch_Read_M_K = np.zeros(
                    [ceil(M / core_tile_M), ceil(K / core_tile_K)], dtype=bool
                )
                previous_batch_Read_K_N = np.zeros(
                    [ceil(K / core_tile_K), ceil(N / core_tile_N)], dtype=bool
                )
                previous_batch_Read_M_N = np.zeros(
                    [ceil(M / core_tile_M), ceil(N / core_tile_N)], dtype=bool
                )
                previous_batch_Write_M_N = np.zeros(
                    [ceil(M / core_tile_M), ceil(N / core_tile_N)], dtype=bool
                )
                previous_batch_compute_cycle_count = 0
                active_core_tile_list = []
                
                for m, n, k in Matmul.generate_tile_loops(
                    ceil(M / core_tile_M),
                    ceil(N / core_tile_N),
                    ceil(K / core_tile_K),
                    Stacked_Mapping.core_loop_order,
                ):
                    active_core_tile_list.append((m, n, k, core_tiles[m, n, k]))
                    
                    # 当累积到 cores_per_channel 个 tile 或到达最后一个 tile 时，处理这一批
                    if (
                        m == ceil(M / core_tile_M) - 1
                        and n == ceil(N / core_tile_N) - 1
                        and k == ceil(K / core_tile_K) - 1
                    ):
                        pass  # 最后一个 tile，强制处理
                    elif len(active_core_tile_list) < cores_per_channel:
                        continue  # 还没凑够一批
                    
                    assert len(active_core_tile_list) <= cores_per_channel
                    
                    current_batch_Read_M_K = np.zeros(
                        [ceil(M / core_tile_M), ceil(K / core_tile_K)], dtype=bool
                    )
                    current_batch_Read_K_N = np.zeros(
                        [ceil(K / core_tile_K), ceil(N / core_tile_N)], dtype=bool
                    )
                    current_batch_Read_M_N = np.zeros(
                        [ceil(M / core_tile_M), ceil(N / core_tile_N)], dtype=bool
                    )
                    current_batch_Write_M_N = np.zeros(
                        [ceil(M / core_tile_M), ceil(N / core_tile_N)], dtype=bool
                    )
                    
                    current_batch_compute_cycle_count = 0
                    for i in range(len(active_core_tile_list)):
                        temp_m, temp_n, temp_k, temp_core_tile = active_core_tile_list[i]
                        current_batch_Read_M_K[temp_m, temp_k] = 1
                        current_batch_Read_K_N[temp_k, temp_n] = 1
                        current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
                        current_batch_Write_M_N[temp_m, temp_n] = 1
                        temp_core_tile_compute_cycle_count = temp_core_tile.compute_cycle_count
                        if temp_k > 0:
                            temp_core_tile_compute_cycle_count += ceil(
                                temp_core_tile.M
                                * temp_core_tile.N
                                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                            )
                        current_batch_compute_cycle_count = max(
                            current_batch_compute_cycle_count,
                            temp_core_tile_compute_cycle_count,
                        )
                    
                    # 计算数据重用后的实际读写量
                    current_batch_M_K_read_count = np.sum(
                        (current_batch_Read_M_K * (~previous_batch_Read_M_K))
                        * M_K_tile_size
                    )
                    current_batch_K_N_read_count = np.sum(
                        (current_batch_Read_K_N * (~previous_batch_Read_K_N))
                        * K_N_tile_size
                    )
                    current_batch_M_N_read_count = np.sum(
                        (
                            current_batch_Read_M_N
                            * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
                        )
                        * M_N_tile_size
                    )
                    previous_batch_M_N_write_count = np.sum(
                        (previous_batch_Write_M_N * (~current_batch_Read_M_N))
                        * M_N_tile_size
                    )
                    # read current batch while compute and write previous batch 流水线处理
                    current_batch_read_count = (
                        current_batch_M_K_read_count
                        + current_batch_K_N_read_count
                        + current_batch_M_N_read_count
                    )
                    current_batch_read_cycle_count = ceil(
                        current_batch_read_count
                        * pcb_module.compute_module.core.systolic_array.input_word_size
                        / pcb_module.compute_module.channel_bandwidth_per_cycle
                    )
                    prvious_batch_write_cycle_count = ceil(
                        previous_batch_M_N_write_count
                        * pcb_module.compute_module.core.systolic_array.output_word_size
                        / pcb_module.compute_module.channel_bandwidth_per_cycle
                    )
                    
                    total_cycle_count += (
                        max(
                            current_batch_read_cycle_count,
                            previous_batch_compute_cycle_count,
                        )
                        + prvious_batch_write_cycle_count
                    )
                    
                    previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                    previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
                    previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
                    previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
                    previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)
                    
                    active_core_tile_list = []
                
                # 最后一批的计算和写回
                total_cycle_count += previous_batch_compute_cycle_count + ceil(
                    np.sum(previous_batch_Write_M_N * M_N_tile_size)
                    * data_type.word_size
                    / pcb_module.compute_module.channel_bandwidth_per_cycle
                )

            return total_cycle_count

    # *Core Tile Simulator需要添加对于输入矩阵的位置的判定，后续考虑接入NoC的模拟器
    class CoreTileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            Stacked_Mapping: "Matmul.Stacked_Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'Core tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.compute_cycle_count = Matmul.simulate_systolic_array_cycle_count(
                look_up_table,
                M,
                N,
                K,
                pcb_module.compute_module.core.systolic_array.array_height,
                pcb_module.compute_module.core.systolic_array.array_width,
                pcb_module.compute_module.core.systolic_array.mac_per_cycle,
                Stacked_Mapping.dataflow,
            )

        def simulate_core_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            Stacked_Mapping: "Matmul.Stacked_Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            assert (
                M * K + K * N + M * N
                <= pcb_module.compute_module.core.SRAM_size
                // data_type.word_size
                // 2
            )

            M_tiling_factor = Stacked_Mapping.l0_M_tiling_factor
            N_tiling_factor = Stacked_Mapping.l0_N_tiling_factor
            K_tiling_factor = Stacked_Mapping.l0_K_tiling_factor
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= pcb_module.compute_module.core.systolic_array_count
            )

            compute_cycle_count = ceil(
                Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    pcb_module.compute_module.core.systolic_array.array_height,
                    pcb_module.compute_module.core.systolic_array.array_width,
                    pcb_module.compute_module.core.systolic_array.mac_per_cycle,
                    Stacked_Mapping.dataflow,
                )
                + (K_tiling_factor - 1)
                * M
                * N
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )
            return compute_cycle_count
        

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        mac_per_clock,
        dataflow="os",
    ):
        # print(f'start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        if M >= array_height and N >= array_width:
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                )
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                )
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
            except KeyError:
                # print('not found in look up table')
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = f"./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )
                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return ceil(cycle_count / mac_per_clock)

    def run_on_gpu(
        self,
    ):
        # import subprocess
        # subprocess.run(['nvidia-smi', '-q', '鈥揹', 'CLOCK'])
        input1 = torch.randn(
            self.computational_graph.M,
            self.computational_graph.K,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        input2 = torch.randn(
            self.computational_graph.K,
            self.computational_graph.N,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        latencies = []
        input1_dummy = torch.ones(4096, 4096).cuda()
        input2_dummy = torch.ones(4096, 4096).cuda()
        # warmup
        for _ in range(3):
            torch.matmul(input1_dummy, input2_dummy)
            torch.cuda.synchronize()
            time.sleep(1)
        for _ in range(self.iterations):
            # x = torch.matmul(input1_dummy, input2_dummy)  # flush the cache
            # torch.cuda.synchronize()
            start = time.time()
            output = torch.matmul(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            assert list(output.shape) == [
                self.computational_graph.M,
                self.computational_graph.N,
            ]
            latencies.append(end - start)
            # time.sleep(1)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # min(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print("GPU kernel launch overhead: ", avg_overhead * 1e3, "ms")
        print(latencies)
        return avg_overhead
