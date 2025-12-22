import math
from typing import List, Tuple, Optional, Dict, Any

from hardware_model.booksim_interface import BookSimInterface, BookSimConfig, TrafficRequest


class NOCModule:
    """
    统一 NoC 抽象：
    - 保存物理参数：bandwidth(B/s), hop_latency(s), freq(Hz), channel_count
    - 保存 BookSim 参数：booksim_path/base_cfg/packet_policy/flit_bytes/...
    - 提供 get_latency(traffic_requests) -> cycles
    """

    def __init__(
        self,
        bandwidth_bps: float,
        hop_latency_s: float,
        channel_count: int,
        freq_hz: float,
        booksim_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.bandwidth = bandwidth_bps
        self.hop_latency = hop_latency_s
        self.channel_count = int(channel_count)
        self.freq_hz = float(freq_hz)

        # mesh 维度（先按正方形）
        k = int(round(math.sqrt(self.channel_count)))
        if k * k != self.channel_count:
            raise ValueError(
                f"[NOCModule] channel_count={self.channel_count} is not a perfect square; "
                f"current Hamiltonian-ring mesh mapping assumes k*k==channel_count."
            )
        if k % 2 != 0:
            raise ValueError(
                f"[NOCModule] mesh_k must be even for Hamiltonian ring mapping, got k={k}."
            )
        self.mesh_k = k

        # 统一映射：logical channel id -> booksim node id(row-major)
        self.logical_to_booksim_id = self._build_hamiltonian_ring_row_major_map(self.mesh_k)

        self._booksim: Optional[BookSimInterface] = None
        self._booksim_cfg: Optional[BookSimConfig] = None

        if booksim_cfg:
            self._booksim_cfg = BookSimConfig(
                booksim_path=booksim_cfg["path"],
                base_cfg_path=booksim_cfg["base_cfg"],
                tmp_dir=booksim_cfg.get("tmp_dir"),
                keep_trace=bool(booksim_cfg.get("keep_trace", False)),
                flit_bytes=booksim_cfg.get("flit_bytes", "auto"),
                packet_policy=booksim_cfg.get("packet_policy", "auto"),
                auto_max_flits=int(booksim_cfg.get("auto_max_flits", 4096)),
                routing_function=booksim_cfg.get("routing_function", "dor"),
                topology=booksim_cfg.get("topology", "mesh"),
                warmup_periods=int(booksim_cfg.get("warmup_periods", 0)),
                max_samples=int(booksim_cfg.get("max_samples", -1)),
            )
            self._booksim = BookSimInterface(self._booksim_cfg)
        self._lut = {}

    @staticmethod
    def _channel_coords_hamiltonian_ring(channel_id: int, mesh_k: int) -> tuple[int, int]:
        """
        与你当前 matmul_HBM.py 的 _get_channel_coords 同构：构建一个哈密顿环，使 (i,i+1) 在物理 mesh 上 1 hop。
        返回 (row=y, col=x)。
        """
        total_nodes = mesh_k * mesh_k
        if channel_id < 0 or channel_id >= total_nodes:
            raise ValueError(f"channel_id out of range: {channel_id}, total={total_nodes}")

        if channel_id == 0:
            return (0, 0)

        # 第一列回路：id in (total_nodes-mesh_k+1 ... total_nodes-1)
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
            out[logical] = y * mesh_k + x  # row-major node id
        return out

    def get_latency(self, traffic_requests, model: str = "booksim") -> int:
        """
        traffic_requests 支持两种格式（delay接口预留，但暂不支持非0）：
        - (src_logical, dst_logical, size_bytes)
        - (delay_cycles, src_logical, dst_logical, size_bytes)  # 仅允许 delay_cycles==0

        返回 cycles（BookSim makespan）。
        """
        model = (model or "booksim").lower()
        if model != "booksim":
            raise ValueError(f"NOCModule.get_latency currently supports model='booksim' only, got {model}")

        if self._booksim is None:
            raise RuntimeError("BookSim is not configured. Provide booksim_cfg in NOCModule init.")

        if not traffic_requests:
            return 0

        # 1) 归一化输入（兼容3/4元组；delay预留但必须为0）
        norm = []
        for req in traffic_requests:
            if len(req) == 3:
                src_ch, dst_ch, size_b = req
                delay = 0
            elif len(req) == 4:
                delay, src_ch, dst_ch, size_b = req
                if int(delay) != 0:
                    raise NotImplementedError(
                        "Delay is reserved in the interface, but non-zero delay is not supported yet."
                    )
            else:
                raise ValueError(f"Invalid traffic request tuple length={len(req)}: {req}")

            src = self.logical_to_booksim_id[int(src_ch)]
            dst = self.logical_to_booksim_id[int(dst_ch)]
            norm.append((src, dst, int(size_b)))  # 当前版本仅支持 delay=0

        # 2) canonicalize（缓存A key）：合并重复(src,dst)，排序后转 tuple
        agg = {}
        for src, dst, size_b in norm:
            key = (src, dst)
            agg[key] = agg.get(key, 0) + size_b

        canon_key = tuple(sorted((src, dst, size_b) for (src, dst), size_b in agg.items()))

        # 3) LUT 命中直接返回
        hit = self._lut.get(canon_key)
        if hit is not None:
            return hit

        # 4) miss：调用 booksim
        cycles = self._booksim.simulate(
            requests=list(canon_key),  # 直接把合并后的 flows 交给 booksim_interface 做拆包
            mesh_k=self.mesh_k,
            bw_bps=self.bandwidth,
            freq_hz=self.freq_hz,
        )

        self._lut[canon_key] = cycles
        return cycles