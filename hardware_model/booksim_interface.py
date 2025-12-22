import os
import re
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Dict


TrafficRequest = Tuple[int, int, int]  # (src_node_id, dst_node_id, size_bytes)


@dataclass
class BookSimConfig:
    booksim_path: str
    base_cfg_path: str
    topology: str = "mesh"
    routing_function: str = "dor"
    n_dim: int = 2
    warmup_periods: int = 0
    max_samples: int = -1
    tmp_dir: Optional[str] = None
    keep_trace: bool = False

    # flit/packet policy
    flit_bytes: str | int = "auto"           # "auto" or int bytes
    packet_policy: str | int = "auto"        # "auto" or int flits per packet
    auto_max_flits: int = 4096               # for auto policy


class BookSimInterface:
    """
    将一组 (src,dst,bytes) 的通信需求转为 BookSim trace + cfg，然后运行 BookSim，解析 makespan(cycles)。

    约定：
    - trace 每行：delay src dst type
      delay 采用“相对延迟”（和你 generate_trace.py 一致）。我们这里全部填 0，表示同周期注入。
    - workload：trace({trace_file},{packet_size_list},-1,0,1)
    - makespan：从 stdout 解析 'Time taken is XXX cycles'
    """

    _TIME_TAKEN_RE = re.compile(r"Time taken is\s+(\d+)\s+cycles", re.IGNORECASE)

    def __init__(self, cfg: BookSimConfig):
        self.cfg = cfg

    @staticmethod
    def _auto_packet_sizes(max_flits: int) -> List[int]:
        # [1,2,4,...,max_flits]
        p = 1
        out = []
        while p <= max_flits:
            out.append(p)
            p <<= 1
        return out

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    @staticmethod
    def _calc_flit_bytes_auto(bw_bps: float, freq_hz: float) -> int:
        # 你已经确认：向上取整即可
        return int(math.ceil(bw_bps / max(1.0, freq_hz)))

    def _resolve_flit_bytes(self, bw_bps: float, freq_hz: float) -> int:
        fb = self.cfg.flit_bytes
        if isinstance(fb, str):
            if fb.lower() != "auto":
                raise ValueError(f"Invalid flit_bytes: {fb}. Use 'auto' or int.")
            return self._calc_flit_bytes_auto(bw_bps, freq_hz)
        return int(fb)

    def _build_trace_lines(
        self,
        requests: List[TrafficRequest],
        flit_bytes: int,
    ) -> Tuple[List[str], List[int]]:
        """
        返回：
        - trace_lines: List[str]
        - packet_size_list: List[int]  # workload 的 {..} 列表，index=type_id
        """
        policy = self.cfg.packet_policy

        if isinstance(policy, str) and policy.lower() == "auto":
            packet_sizes = self._auto_packet_sizes(self.cfg.auto_max_flits)
            size_to_type: Dict[int, int] = {sz: i for i, sz in enumerate(packet_sizes)}

            trace_lines: List[str] = []
            for item in requests:
                if len(item) == 3:
                    src, dst, size_bytes = item
                    delay = 0
                elif len(item) == 4:
                    delay, src, dst, size_bytes = item
                    if int(delay) != 0:
                        raise NotImplementedError(
                            "Delay is reserved in the interface, but non-zero delay is not supported yet."
                        )
                else:
                    raise ValueError(f"Invalid request tuple length={len(item)}: {item}")

                total_flits = int(math.ceil(size_bytes / max(1, flit_bytes)))

                # 二进制分解：从大到小
                remaining = total_flits
                for sz in reversed(packet_sizes):
                    cnt = remaining // sz
                    if cnt <= 0:
                        continue
                    type_id = size_to_type[sz]
                    # 这里 delay 都填 0：同周期注入
                    trace_lines.extend([f"0 {src} {dst} {type_id}\n"] * cnt)
                    remaining -= cnt * sz
                    if remaining == 0:
                        break
                if remaining != 0:
                    raise RuntimeError(f"Auto packet decomposition failed: remaining={remaining}")

            return trace_lines, packet_sizes

        # fixed flits-per-packet
        fixed = int(policy)
        if fixed <= 0:
            raise ValueError(f"Invalid packet_policy={policy}. Use 'auto' or positive int.")
        trace_lines = []
        for item in requests:
            if len(item) == 3:
                src, dst, size_bytes = item
                delay = 0
            elif len(item) == 4:
                delay, src, dst, size_bytes = item
                if int(delay) != 0:
                    raise NotImplementedError(
                        "Delay is reserved in the interface, but non-zero delay is not supported yet."
                    )
            else:
                raise ValueError(f"Invalid request tuple length={len(item)}: {item}")

            total_flits = int(math.ceil(size_bytes / max(1, flit_bytes)))
            # fixed 模式允许 overshoot（最后一个包仍按 fixed 记）
            num_packets = self._ceil_div(total_flits, fixed)
            trace_lines.extend([f"0 {src} {dst} 0\n"] * num_packets)

        return trace_lines, [fixed]

    def _run_booksim(self, cfg_path: str, extra_args: List[str]) -> str:
        if not os.path.exists(self.cfg.booksim_path):
            raise FileNotFoundError(f"BookSim not found: {self.cfg.booksim_path}")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"BookSim base cfg not found: {cfg_path}")

        cmd = [self.cfg.booksim_path, cfg_path] + extra_args
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (result.stdout or "") + "\n" + (result.stderr or "")

        # 关键修改：允许“非0返回码但输出里有 Time taken”的情况继续走
        if result.returncode != 0:
            if self._TIME_TAKEN_RE.search(out):
                # 可选：打印一条警告，方便你之后追 rc=255 的来源
                # print(f"[BookSim] warning: non-zero return code {result.returncode}, but got Time taken; treating as success")
                return out
            raise RuntimeError(f"BookSim failed (rc={result.returncode}). Output:\n{out}")

        return out

    def _parse_time_taken_cycles(self, output: str) -> int:
        m = self._TIME_TAKEN_RE.search(output)
        if not m:
            raise RuntimeError(f"Cannot find 'Time taken is ... cycles' in BookSim output.\n{output[:2000]}")
        return int(m.group(1))

    def simulate(
        self,
        requests: List[TrafficRequest],
        mesh_k: int,
        bw_bps: float,
        freq_hz: float,
    ) -> int:
        """
        返回 makespan cycles（comm_cycles）。
        """
        if not requests:
            return 0

        flit_bytes = self._resolve_flit_bytes(bw_bps, freq_hz)
        trace_lines, packet_sizes = self._build_trace_lines(requests, flit_bytes)

        # workload 参数：trace({trace_file},{packet_sizes},-1,0,1)
        packet_sizes_str = "{" + ",".join(str(x) for x in packet_sizes) + "}"

        # 临时目录
        if self.cfg.tmp_dir:
            os.makedirs(self.cfg.tmp_dir, exist_ok=True)
            tmp_ctx = tempfile.TemporaryDirectory(dir=self.cfg.tmp_dir)
        else:
            tmp_ctx = tempfile.TemporaryDirectory()

        with tmp_ctx as td:
            trace_path = os.path.join(td, "trace.txt")
            with open(trace_path, "w") as f:
                f.writelines(trace_lines)

            # 使用 base_cfg + overrides（不改模板文件，避免污染）
            # workload_arg = f"- workload=\"trace({{{trace_path}}},{packet_sizes_str},-1,0,1)\""
            # extra_args = [
            #     "- sim_type=workload",
            #     f"- topology={self.cfg.topology}",
            #     f"- k={mesh_k}",
            #     f"- n={self.cfg.n_dim}",
            #     f"- routing_function={self.cfg.routing_function}",
            #     workload_arg,
            #     f"- warmup_periods={self.cfg.warmup_periods}",
            #     f"- max_samples={self.cfg.max_samples}",
            # ]
            # print("extra_args = ", extra_args)
            packet_sizes_str = "{" + ",".join(str(x) for x in packet_sizes) + "}"
            workload_val = f"workload=trace({{{trace_path}}},{packet_sizes_str},-1,0,1)"

            extra_args = [
                "-", "sim_type=workload",
                "-", f"topology={self.cfg.topology}",
                "-", f"k={mesh_k}",
                "-", f"n={self.cfg.n_dim}",
                "-", f"routing_function={self.cfg.routing_function}",
                "-", workload_val,
                "-", f"warmup_periods={self.cfg.warmup_periods}",
                "-", f"max_samples={self.cfg.max_samples}",
            ]

            output = self._run_booksim(self.cfg.base_cfg_path, extra_args)
            cycles = self._parse_time_taken_cycles(output)

            if self.cfg.keep_trace:
                # 复制到 tmp_dir 之外方便你对照调试
                keep_dir = self.cfg.tmp_dir or "."
                os.makedirs(keep_dir, exist_ok=True)
                shutil.copy(trace_path, os.path.join(keep_dir, "last_booksim_trace.txt"))

            return cycles