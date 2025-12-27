import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hardware_model.booksim_interface import BookSimInterface, BookSimConfig

# 统一请求格式（delay 预留：支持 3/4 元组，但当前只允许 delay=0）
# (src_logical, dst_logical, bytes)
# (delay, src_logical, dst_logical, bytes)  # delay!=0 => NotImplementedError
TrafficReqIn = Tuple[int, ...]  # 兼容 3/4 元组
Flow = Tuple[int, int, int]     # (src_node_id, dst_node_id, bytes_sum)


class NoCModel:
    def comm_cycles(self, flows: List[Flow], noc: "NOCModule") -> int:
        raise NotImplementedError


class AnalyticalNoCModel(NoCModel):
    """
    Wormhole（零负载）估算：每条 flow 的时间 = hops*hop_latency + bytes/bw
    makespan 取 max(flow_time)
    """
    def comm_cycles(self, flows: List[Flow], noc: "NOCModule") -> int:
        if not flows:
            return 0

        worst = 0
        for src, dst, size_b in flows:
            hops = noc.manhattan_hops(src, dst)
            t_sec = hops * noc.hop_latency_s + (size_b / noc.bandwidth_bps)
            cyc = int(math.ceil(t_sec * noc.freq_hz))
            if cyc > worst:
                worst = cyc
        return worst


class BookSimNoCModel(NoCModel):
    """
    BookSim 精确模拟（含运行内缓存A/LUT）
    key = tuple(sorted((src,dst,bytes_sum))) after mapping+aggregation
    """
    def __init__(self, bs: BookSimInterface):
        self.bs = bs
        self._lut: Dict[Tuple[Tuple[int, int, int], ...], int] = {}

    def comm_cycles(self, flows: List[Flow], noc: "NOCModule") -> int:
        if not flows:
            return 0
        key = tuple(sorted(flows))
        hit = self._lut.get(key)
        if hit is not None:
            return hit

        cycles = self.bs.simulate(
            requests=list(key),
            mesh_k=noc.mesh_k,
            bw_bps=noc.bandwidth_bps,
            freq_hz=noc.freq_hz,
        )
        self._lut[key] = cycles
        return cycles


@dataclass
class NOCModule:
    bandwidth_bps: float
    hop_latency_s: float
    channel_count: int
    freq_hz: float
    model: str = "estimate"
    booksim_cfg: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.channel_count = int(self.channel_count)
        self.freq_hz = float(self.freq_hz)
        self.bandwidth_bps = float(self.bandwidth_bps)
        self.hop_latency_s = float(self.hop_latency_s)

        # mesh 维度（按正方形；你当前 channel_count=16 => k=4）
        k = int(round(math.sqrt(self.channel_count)))
        if k * k != self.channel_count:
            raise ValueError(
                f"[NOCModule] channel_count={self.channel_count} is not a perfect square; "
                f"current Hamiltonian-ring mapping assumes k*k==channel_count."
            )
        if k % 2 != 0:
            raise ValueError(
                f"[NOCModule] mesh_k must be even for Hamiltonian ring mapping, got k={k}."
            )
        self.mesh_k = k

        # logical channel id -> booksim node id(row-major)
        self.logical_to_booksim_id = self._build_hamiltonian_ring_row_major_map(self.mesh_k)

        # 选择模型（像 DRAMModel 一样）
        m = self.model.lower()
        if m == "estimate":
            self._impl = AnalyticalNoCModel()
        elif m == "booksim":
            if not self.booksim_cfg:
                raise ValueError("[NOCModule] noc_model=booksim but booksim_cfg is missing")
            bs_cfg = BookSimConfig(
                booksim_path=self.booksim_cfg["path"],
                base_cfg_path=self.booksim_cfg["base_cfg"],
                tmp_dir=self.booksim_cfg.get("tmp_dir"),
                keep_trace=bool(self.booksim_cfg.get("keep_trace", False)),
                flit_bytes=self.booksim_cfg.get("flit_bytes", "auto"),
                packet_policy=self.booksim_cfg.get("packet_policy", "auto"),
                auto_max_flits=int(self.booksim_cfg.get("auto_max_flits", 4096)),
                routing_function=self.booksim_cfg.get("routing_function", "dor"),
                topology=self.booksim_cfg.get("topology", "mesh"),
                warmup_periods=int(self.booksim_cfg.get("warmup_periods", 0)),
                max_samples=int(self.booksim_cfg.get("max_samples", -1)),
            )
            self._impl = BookSimNoCModel(BookSimInterface(bs_cfg))
        else:
            raise ValueError(f"[NOCModule] unknown noc model: {self.model}")

    @staticmethod
    def _channel_coords_hamiltonian_ring(channel_id: int, mesh_k: int) -> tuple[int, int]:
        total_nodes = mesh_k * mesh_k
        if channel_id < 0 or channel_id >= total_nodes:
            raise ValueError(f"channel_id out of range: {channel_id}, total={total_nodes}")

        if channel_id == 0:
            return (0, 0)

        # 第一列回路
        if channel_id > total_nodes - mesh_k:
            row = total_nodes - channel_id
            return (row, 0)

        effective_id = channel_id - 1
        sub_grid_width = mesh_k - 1
        row = effective_id // sub_grid_width
        col_offset = effective_id % sub_grid_width
        if row % 2 == 0:
            col = 1 + col_offset
        else:
            col = sub_grid_width - col_offset
        return (row, col)

    @classmethod
    def _build_hamiltonian_ring_row_major_map(cls, mesh_k: int) -> List[int]:
        total = mesh_k * mesh_k
        out = [0] * total
        for logical in range(total):
            y, x = cls._channel_coords_hamiltonian_ring(logical, mesh_k)
            out[logical] = y * mesh_k + x
        return out

    def _normalize_requests(self, traffic_requests: List[TrafficReqIn]) -> List[Tuple[int, int, int]]:
        """
        输出规范化后的 (src_node_id, dst_node_id, bytes)
        delay 预留：允许传 4 元组，但 delay!=0 直接报错，避免“看起来支持但实际上被忽略”。
        """
        out: List[Tuple[int, int, int]] = []
        for req in traffic_requests:
            if len(req) == 3:
                src_ch, dst_ch, size_b = req
                delay = 0
            elif len(req) == 4:
                delay, src_ch, dst_ch, size_b = req
                if int(delay) != 0:
                    raise NotImplementedError("Delay is reserved but non-zero delay is not supported yet.")
            else:
                raise ValueError(f"Invalid traffic request tuple length={len(req)}: {req}")

            src = self.logical_to_booksim_id[int(src_ch)]
            dst = self.logical_to_booksim_id[int(dst_ch)]
            out.append((src, dst, int(size_b)))
        return out

    def _aggregate_flows(self, reqs: List[Tuple[int, int, int]]) -> List[Flow]:
        agg: Dict[Tuple[int, int], int] = {}
        for src, dst, b in reqs:
            agg[(src, dst)] = agg.get((src, dst), 0) + b
        return [(src, dst, b) for (src, dst), b in agg.items()]

    def manhattan_hops(self, src_node: int, dst_node: int) -> int:
        y1, x1 = divmod(src_node, self.mesh_k)
        y2, x2 = divmod(dst_node, self.mesh_k)
        return abs(y1 - y2) + abs(x1 - x2)

    def get_latency(self, traffic_requests: List[TrafficReqIn]) -> int:
        """
        统一入口：调用方只传 requests，模型由 config 决定。
        """
        if not traffic_requests:
            return 0
        reqs = self._normalize_requests(traffic_requests)
        flows = self._aggregate_flows(reqs)
        return self._impl.comm_cycles(flows, self)