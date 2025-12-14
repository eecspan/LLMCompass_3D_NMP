import math
from typing import Optional, Tuple, Dict


class DRAMModel:
    def load_cycles(self, size_bytes: int, word_size: int = 2) -> int:
        raise NotImplementedError
    def store_cycles(self, size_bytes: int, word_size: int = 2) -> int:
        raise NotImplementedError


class BandwidthDRAMModel(DRAMModel):
    def __init__(self, bandwidth_per_cycle: float):
        # 单通道等效带宽（B/cycle），或“总带宽/并行度”的等效值
        self.bandwidth_per_cycle = bandwidth_per_cycle
    def _cycles(self, size_bytes: int) -> int:
        return int(math.ceil(size_bytes / max(1.0, self.bandwidth_per_cycle)))
    def load_cycles(self, size_bytes: int, word_size: int = 2) -> int:
        return self._cycles(size_bytes)
    def store_cycles(self, size_bytes: int, word_size: int = 2) -> int:
        return self._cycles(size_bytes)


class RamulatorDRAMModel(DRAMModel):
    def __init__(self, channel_count: int, ramulator_path: str, clean_temp: bool = False, cfg: Optional[Dict] = None):
        from hardware_model import ramulator_interface as rami
        self.rami = rami
        cfg = dict(cfg or {})
        self.rami.apply_config(cfg)  # 平铺键值：n_pch/n_rank/.../data_size 等
        self.ramulator_path = ramulator_path
        self.clean_temp = clean_temp
        self._lut: Dict[Tuple[str, int, int], int] = {}

    def _get_cycles(self, op: str, size_bytes: int, word_size: int) -> int:
        key = (op, size_bytes, word_size)
        hit = self._lut.get(key)
        if hit is not None:
            return hit
        elems = int(math.ceil(size_bytes / max(1, word_size)))
        inst_type = 0 if op == "LD" else 1
        # 以 (fcN=1, fcK=elems) 近似顺序访问；若需精准模式可扩展这里的访问模式
        cycles = self.rami.load_store_cycle_func(
            inst_type=inst_type,
            fcN=1,
            fcK=elems,
            data_size=word_size,
            ramulator_path=self.ramulator_path,
            clean_temp=self.clean_temp,
        )
        self._lut[key] = cycles
        return cycles

    def load_cycles(self, size_bytes: int, word_size: int = 2) -> int:
        return self._get_cycles("LD", size_bytes, word_size)

    def store_cycles(self, size_bytes: int, word_size: int = 2) -> int:
        return self._get_cycles("ST", size_bytes, word_size)


class MemoryModule:
    def __init__(self, memory_capacity, channel_count, dram_model: Optional[DRAMModel] = None):
        self.memory_capacity = memory_capacity
        self.channel_count = channel_count  # Number of memory channels
        self.dram_model = dram_model

    def set_dram_model(self, dram_model: DRAMModel):
        self.dram_model = dram_model


memory_module_dict = {
    'A100_80GB': MemoryModule(80e9, channel_count=40),
    'TPUv3': MemoryModule(float('inf'),channel_count=1),
    'MI210': MemoryModule(64e9, channel_count=32)
}