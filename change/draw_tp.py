import numpy as np
import math
import heapq
from collections import deque

# =================================================================================
# 1. 配置模块 (已实现PE Array形状推导开关)
# =================================================================================
class HardwareConfig:
    def __init__(self, core_flops: float, core_bw: float, noc_bw: float, 
                 noc_first_latency: float, noc_hop_latency: float, 
                 num_cores: int, core_dim_x: int, core_dim_y: int,
                 frequency_hz: float = 1e9,
                 pe_shape_mode: str = 'intensity_aware',
                 data_type_bytes: int = 2):
        
        # 用户提供的宏观和网络参数
        self.target_core_flops = core_flops
        self.core_bw = core_bw
        self.noc_bw, self.noc_first_latency, self.noc_hop_latency = noc_bw, noc_first_latency, noc_hop_latency
        self.num_cores, self.core_dim_x, self.core_dim_y = num_cores, core_dim_x, core_dim_y
        self.data_type_bytes = data_type_bytes
        self.frequency_hz = frequency_hz
        self.pe_shape_mode = pe_shape_mode

        if self.frequency_hz <= 0 or self.core_bw <=0:
            raise ValueError("频率和带宽必须为正数")
            
        total_pe_elements = self.target_core_flops / (2 * self.frequency_hz)

        if self.pe_shape_mode == 'square':
            # 模式1: 推导方形PE Array
            ideal_dim = math.sqrt(total_pe_elements)
            if ideal_dim == 0:
                self.pe_array_dim_m = self.pe_array_dim_n = 0
            else:
                power_of_2 = int(math.pow(2, round(math.log2(ideal_dim))))
                self.pe_array_dim_m = self.pe_array_dim_n = max(1, power_of_2)

        elif self.pe_shape_mode == 'intensity_aware':
            # 模式2: 推导计算强度自适应的非对称PE Array
            machine_intensity = self.target_core_flops / self.core_bw
            if machine_intensity < 1:
                self.pe_array_dim_m = 16 # 对访存密集型硬件，M维不宜过大
            else:
                self.pe_array_dim_m = int(math.pow(2, round(math.log2(machine_intensity))))
            # self.pe_array_dim_m = 10
            self.pe_array_dim_m = max(1, self.pe_array_dim_m)
            
            ideal_dim_n = total_pe_elements / self.pe_array_dim_m
            if ideal_dim_n < 1:
                self.pe_array_dim_n = 1
            else:
                self.pe_array_dim_n = int(math.pow(2, round(math.log2(ideal_dim_n))))
                self.pe_array_dim_n = ideal_dim_n
        
        else:
            raise ValueError("pe_shape_mode 必须是 'square' 或 'intensity_aware'")

        # 使用推导出的维度，重新计算一个“名义算力”以保证内部自洽
        self.core_flops = self.pe_array_dim_m * self.pe_array_dim_n * 2 * self.frequency_hz
        
        # 其他参数
        self.noc_size_per_cycle = noc_bw * noc_hop_latency
        if self.noc_size_per_cycle == 0: self.noc_size_per_cycle = 1e-12
        if self.num_cores != self.core_dim_x * self.core_dim_y: raise ValueError("核心数与维度不匹配")

class MoEModelConfig:
    def __init__(self, hidden_size: int, moe_intermediate_size: int, num_experts: int):
        self.hidden_size, self.moe_intermediate_size, self.num_experts = hidden_size, moe_intermediate_size, num_experts


# =================================================================================
# 2. 模拟器核心模块 (已进行物理感知修正)
# =================================================================================

class MoESimulator:
    def __init__(self, hw_config: HardwareConfig, model_config: MoEModelConfig, comm_model: str = 'hierarchical', compute_model: str = 'fine_grained'):
        self.hw = hw_config
        self.model = model_config
        self.comm_model = comm_model
        self.compute_model = compute_model
        
        if self.comm_model not in ['ring', '2d_mesh', 'hierarchical']:
            raise ValueError("通信模型必须是 'ring', '2d_mesh', 或 'hierarchical'")
        if self.compute_model not in ['coarse', 'fine_grained']:
            raise ValueError("计算模型必须是 'coarse' 或 'fine_grained'")
            
        print(f"模拟器初始化成功！计算模型: {self.compute_model}, 推导模式: '{self.hw.pe_shape_mode}'")
        print(f"  目标算力: {self.hw.target_core_flops/1e12:.2f} TFLOPS, 带宽: {self.hw.core_bw/1e9} GB/s -> "
              f"推断 PE Array 维度: {self.hw.pe_array_dim_m}x{self.hw.pe_array_dim_n}")
        print(f"  校准后名义算力: {self.hw.core_flops/1e12:.2f} TFLOPS")


    def _get_coords(self, core_id: int) -> tuple[int, int]:
        if core_id >= self.hw.num_cores: raise ValueError(f"核心ID {core_id} 超出范围")
        return (core_id // self.hw.core_dim_x, core_id % self.hw.core_dim_x)

    def _get_manhattan_distance(self, id1: int, id2: int) -> int:
        y1, x1 = self._get_coords(id1)
        y2, x2 = self._get_coords(id2)
        return abs(y1 - y2) + abs(x1 - x2)

    def _get_step_time_components(self, M_dim_effective: int) -> tuple[float, float]:
        tile_m, tile_n = self.hw.pe_array_dim_m, self.hw.pe_array_dim_n
        tile_k = tile_n
        if tile_m * tile_n * tile_k == 0: return float('inf'), float('inf')
        # t_compute = tile_k / self.hw.frequency_hz
        t_compute = tile_m * tile_k * tile_n * 2 / self.hw.core_flops
        bytes_to_load = (M_dim_effective * tile_k + tile_k * tile_n) * self.hw.data_type_bytes
        t_memory = bytes_to_load / self.hw.core_bw
        return t_compute, t_memory

    def _get_expert_comp_load_time(self, token_num: int, num_cores_tp: int = 1) -> tuple[float, float]:
        if self.compute_model == 'coarse':
            return self._get_expert_time_coarse(token_num, num_cores_tp)
        else:
            return self._get_expert_time_fine_grained(token_num, num_cores_tp)

    def _get_expert_time_coarse(self, token_num: int, num_cores_tp: int = 1) -> tuple[float, float]:
        if token_num == 0: return 0.0, 0.0
        H, I = self.model.hidden_size, self.model.moe_intermediate_size
        total_flops = (token_num * H * I * 2 + token_num * I * H) * 2
        total_weight_bytes = (H * I * 2 + I * H) * self.hw.data_type_bytes
        total_input_bytes = token_num * H * 3 * self.hw.data_type_bytes
        effective_flops = self.hw.core_flops * num_cores_tp
        effective_bw = self.hw.core_bw * num_cores_tp
        time_compute = total_flops / effective_flops
        time_memory = (total_weight_bytes + total_input_bytes) / effective_bw
        return max(time_compute, time_memory), time_compute


    def _get_mlp_time_fine_grained(self, M: int, K: int, N: int, num_cores_tp: int = 1) -> float:
        if M * K * N == 0: return 0.0
        tile_m, tile_n = self.hw.pe_array_dim_m, self.hw.pe_array_dim_n
        tile_k = tile_n
        if tile_m * tile_n * tile_k == 0: return float('inf')
        t_compute_per_step = tile_m * tile_k * tile_n * 2 / self.hw.core_flops
        effective_tile_m = min(M, tile_m)
        bytes_to_load_per_step = (effective_tile_m * tile_k + tile_k * tile_n) * self.hw.data_type_bytes
        effective_bw = self.hw.core_bw * num_cores_tp
        t_memory_per_step = bytes_to_load_per_step / effective_bw
        time_per_step = max(t_compute_per_step, t_memory_per_step)
        t_effective_compute_per_step = t_compute_per_step * effective_tile_m / tile_m
        effective_N = math.ceil(N / num_cores_tp)
        num_steps_m, num_steps_k, num_steps_n = math.ceil(M / tile_m), math.ceil(K / tile_k), math.ceil(effective_N / tile_n)
        total_steps = num_steps_m * num_steps_k * num_steps_n
        return total_steps * time_per_step, total_steps * t_effective_compute_per_step

    def _get_expert_time_fine_grained(self, token_num: int, num_cores_tp: int = 1) -> tuple[float, float]:
        H, I = self.model.hidden_size, self.model.moe_intermediate_size
        time_up, time_compute_up = self._get_mlp_time_fine_grained(token_num, H, I, num_cores_tp)
        time_gate, time_compute_gate = self._get_mlp_time_fine_grained(token_num, H, I, num_cores_tp)
        time_down, time_compute_down = self._get_mlp_time_fine_grained(token_num, I, H, num_cores_tp)
        total_time = time_up + time_gate + time_down
        total_time_compute = time_compute_up + time_compute_gate + time_compute_down
        return total_time, total_time_compute

    def _get_step_time(self, M_dim_effective: int) -> float:
        tile_m, tile_n = self.hw.pe_array_dim_m, self.hw.pe_array_dim_n
        tile_k = tile_n
        if tile_m * tile_n * tile_k == 0: return float('inf')
        
        # t_compute_per_step = tile_k / self.hw.frequency_hz
        t_compute_per_step = tile_m * tile_k * tile_n * 2 / self.hw.core_flops
        bytes_to_load_per_step = (M_dim_effective * tile_k + tile_k * tile_n) * self.hw.data_type_bytes
        t_memory_per_step = bytes_to_load_per_step / self.hw.core_bw
        return max(t_compute_per_step, t_memory_per_step)

    def _get_all_reduce_time(self, data_size_bytes: int, num_cores_in_group: int) -> float:
        """根据配置的通信模型计算All-Reduce时间（已进行物理感知修正）"""
        if num_cores_in_group <= 1 or data_size_bytes == 0:
            return 0.0

        if self.comm_model == 'ring':
            # 【已修正】Ring All-Reduce，每步通信的跳数不再是1，而是实际物理距离
            total_latency = 0.0
            chunk_size = data_size_bytes / num_cores_in_group
            num_packets_per_chunk = chunk_size / self.hw.noc_size_per_cycle
            transmission_time = max(0, num_packets_per_chunk - 1) * self.hw.noc_hop_latency
            
            # 模拟2*(N-1)个步骤
            for step in range(1, num_cores_in_group):
                # 在每个步骤中，所有核心i都向逻辑邻居发送数据
                # 这里我们计算一次代表性的传输延迟即可
                # Reduce-Scatter阶段：核心i发送给(i + step) % N
                # All-Gather阶段：核心i发送给(i + step) % N
                # 我们简化计算，假设每一步所有传输的平均距离近似于一个核心与其逻辑邻居的距离
                # 核心i的邻居是 (i-1+N)%N 和 (i+1+N)%N
                # 我们以核心0到核心1的距离为例，这在环上是代表性的
                # 注意：一个更精确的模型会追踪每个数据块的传递路径，但这里我们做一个合理的简化
                sender = 0
                receiver = 1
                hops = self._get_manhattan_distance(sender, receiver)
                
                # 计算单步通信时间
                step_latency = hops * self.hw.noc_first_latency + transmission_time
                total_latency += step_latency
            
            # Ring All-Reduce 有两个大阶段 (Reduce-Scatter 和 All-Gather)
            return total_latency * 2

        elif self.comm_model == 'hierarchical':
            # —— 无回环长边 的分层 2D（邻居-only，两次扫）——
            if num_cores_in_group <= 1 or data_size_bytes == 0:
                return 0.0

            # 维度分解：N = dim_x * dim_y
            dim_y = int(np.sqrt(num_cores_in_group))
            while num_cores_in_group % dim_y != 0:
                dim_y -= 1
            dim_x = num_cores_in_group // dim_y

            B  = self.hw.noc_size_per_cycle         # 每拍可传字节
            Lf = self.hw.noc_first_latency          # 首跳/启动延迟
            Lh = self.hw.noc_hop_latency            # 每额外包的流水吞吐延迟

            import math
            def _packets(sz_bytes: int) -> int:
                # 分包数：虫孔模型中用于计算串行项
                return max(0, math.ceil(sz_bytes / B))

            def _step_time(sz_bytes: int, hops: int = 1) -> float:
                """
                单次“邻居交换”的时延：
                启动项：hops * Lf
                串行项：(pkts - 1) * Lh   （注意：不乘 hops）
                """
                pkts = _packets(sz_bytes)
                if pkts <= 1:
                    return hops * Lf
                return hops * Lf + (pkts - 1) * Lh

            # 相邻平均跳数：若行/列映射连续则为 1；如需更精确可替换为统计平均物理距离
            H_row = 1
            H_col = 1

            # 行内 RS 后的分片大小
            slice_size = data_size_bytes / dim_x

            # 1) 行内 RS：两次扫（左->右 前缀规约 + 右->左 后缀补齐并“就地散片”）
            t_row_rs = 2 * max(0, dim_x - 1) * _step_time(slice_size, H_row)

            # 2) 列内 All-Reduce = 列 RS（两次扫）+ 列 AG（两次扫），始终在 slice_size 上进行
            t_col_rs = 2 * max(0, dim_y - 1) * _step_time(slice_size, H_col)
            t_col_ag = 2 * max(0, dim_y - 1) * _step_time(slice_size, H_col)

            # 3) 行内 AG：两次扫（左<->右），在 slice_size 上拼回完整 S
            t_row_ag = 2 * max(0, dim_x - 1) * _step_time(slice_size, H_row)

            return t_row_rs + t_col_rs + t_col_ag + t_row_ag

        else: # '2d_mesh'
            num_packets = data_size_bytes / self.hw.noc_size_per_cycle
            dim_y = int(np.sqrt(num_cores_in_group))
            while num_cores_in_group % dim_y != 0: dim_y -= 1
            dim_x = num_cores_in_group // dim_y
            
            def op_latency(dim):
                if dim <= 1: return 0.0
                hops = dim - 1 # 沿着物理行/列移动，总跳数是dim-1
                startup_time = hops * self.hw.noc_first_latency
                transmission_time = max(0, num_packets - 1) * self.hw.noc_hop_latency
                return startup_time + transmission_time
            
            return 2 * (op_latency(dim_x) + op_latency(dim_y))

    def _get_point_to_point_comm_time(self, size_bytes: int, source_core_id: int, dest_core_id: int) -> float:
        """
        【新增】计算一次点对点通信的精确延迟，考虑首跳延迟。
        模型: T = (hops * Lf) + (packets - 1) * Lh
        """
        if size_bytes <= 0: return 0.0
        B  = self.hw.noc_size_per_cycle
        Lf = self.hw.noc_first_latency
        Lh = self.hw.noc_hop_latency
        hops = self._get_manhattan_distance(source_core_id, dest_core_id)
        num_packets = math.ceil(size_bytes / B)
        if num_packets <= 1: return hops * Lf
        startup_latency = hops * Lf
        serialization_latency = (num_packets - 1) * Lh
        return startup_latency + serialization_latency


# deepseek expert配置
hidden_size=2048
moe_intermediate_size=1408
num_experts=64

# qwen
# hidden_size=2048
# moe_intermediate_size = 768
# num_experts=128

# hidden_size=3072
# moe_intermediate_size = 8192
# num_experts=8

cores = [4, 8, 16, 32, 64, 128]
core_dim_xs = [2, 2, 4, 4, 8, 8]  # 假设每个配置的核心都是方形或近似方形
core_dim_ys = [2, 4, 4, 8, 8, 16]  # 假设每个配置的核心都是方形或近似方形

# 128核的硬件配置
TARGET_FLOPS = 672e9  # 8 TFLOPS
TARGET_BW = 84e9   # 256 GB/s
PE_ARRAY_SIZE = 'intensity_aware'  # 可选 'square' 或 'intensity_aware'

model_config = MoEModelConfig(hidden_size=hidden_size, moe_intermediate_size=moe_intermediate_size, num_experts=num_experts)
    
computation_times = []
noc_times = []
for _, core in enumerate(cores):
    core_dim_x = core_dim_xs[_]
    core_dim_y = core_dim_ys[_]
    hw_config = HardwareConfig(
        core_flops=TARGET_FLOPS, core_bw=TARGET_BW, noc_bw=TARGET_BW,
        noc_first_latency=25e-9, noc_hop_latency=1e-9,
        num_cores=core, core_dim_x=core_dim_x, core_dim_y=core_dim_y, pe_shape_mode=PE_ARRAY_SIZE,)

    simulator = MoESimulator(hw_config, model_config, comm_model='hierarchical', compute_model='coarse')

    token_num = 16
    N = simulator.hw.num_cores
    comp_load_time, comp_time = simulator._get_expert_comp_load_time(token_num, N)
    comm_time = simulator._get_all_reduce_time(token_num * simulator.model.hidden_size * simulator.hw.data_type_bytes, N)
    computation_times.append(comp_load_time)
    noc_times.append(comm_time)


import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import PercentFormatter # 导入百分比格式化工具

# --- 1. 定义实验配置 ---
# X轴：不同的核心数量
# 颜色
line_color = '#ee6677' # 沿用之前代表“开销”的警示红/橙色

# --- 2. 原始伪数据 (单位: ms) ---
time_data = {
    'Computation Time':       np.array(computation_times),
    'NoC Communication Time': np.array(noc_times)
}

# --- 3. 【核心】计算 “NoC时间占比” ---
total_time = time_data['Computation Time'] + time_data['NoC Communication Time']
noc_proportion = (time_data['NoC Communication Time'] / total_time) * 100 # 转换为百分比

# --- 4. 开始绘图 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
# 使用一个紧凑的画布尺寸
fig, ax = plt.subplots(figsize=(8, 3))

# 绘制折线图
ax.plot(cores, noc_proportion, 
        color=line_color, 
        marker='o',       # 在每个数据点上添加标记
        markersize=8,
        linewidth=2.5,
        label='NoC Overhead Percentage',
        markerfacecolor='white',
        markeredgewidth=2)

# --- 5. 美化和格式化图表 ---
# 设置标题和坐标轴标签
# ax.set_title('Dominance of NoC Overhead with Increasing Scale', fontsize=14, pad=15)
ax.set_xlabel('Number of Cores', fontsize=18)
ax.set_ylabel('NoC overhead (%)', fontsize=18)

# 设置坐标轴
ax.tick_params(axis='both', which='major', labelsize=14)
# 将X轴设置为以2为底的对数刻度
ax.set_xscale('log', base=2)
# 确保X轴只显示我们有的数据点
ax.set_xticks(cores)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # 以常规数字显示标签
# 将Y轴设置为0-100%
ax.set_ylim(0, 100)

# 添加网格线
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

# 移除顶部和右侧的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 在每个数据点旁边标注精确的百分比
for i, percentage in enumerate(noc_proportion):
    ax.text(cores[i] * 1.1, percentage, f'{percentage:.1f}%', 
            va='center', fontsize=14, fontweight='bold', color=line_color)

# 调整整体布局
plt.tight_layout()

# --- 6. 保存 ---
output_filename = "noc_percentage_trend.pdf"
plt.savefig(output_filename, format='pdf')
plt.show()

print(f"NoC时间占比折线图已成功保存为 '{os.path.abspath(output_filename)}'")