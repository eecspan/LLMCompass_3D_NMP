import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

from software_model.utils import data_type_dict, DataType
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)


@dataclass
class ModelConfig:
    name: Optional[str]
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    data_type: DataType = data_type_dict["fp16"]
    device_count: Optional[int] = None


def read_model_template(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def template_to_model_config(
    specs: Dict[str, Any],
    default_device_count: Optional[int] = None,
) -> ModelConfig:

    dtype_name = specs.get("data_type", "fp16")
    dtype = data_type_dict.get(dtype_name, data_type_dict["fp16"])

    cfg = ModelConfig(
        name=specs.get("name") or specs.get("model_type"),
        hidden_size=int(specs["hidden_size"]),
        num_attention_heads=int(specs["num_attention_heads"]),
        num_hidden_layers=int(specs["num_hidden_layers"]),
        num_key_value_heads=specs.get("num_key_value_heads"),
        vocab_size=specs.get("vocab_size"),
        intermediate_size=specs.get("intermediate_size"),
        data_type=dtype,
        device_count=specs.get("device_count", default_device_count),
    )

    # 基础校验
    if cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError(f"hidden_size ({cfg.hidden_size}) 必须能整除 num_attention_heads ({cfg.num_attention_heads}).")
    return cfg


def build_blocks_from_config(cfg: ModelConfig, system=None):
    """
    返回 (prefill_block, decode_block)，device_count 优先取 cfg.device_count，
    若未指定则回落为 system.interconnect.device_count，最后回落为 1。
    """
    if cfg.device_count is not None:
        dev_cnt = int(cfg.device_count)
    elif system is not None and hasattr(system, "interconnect"):
        dev_cnt = int(getattr(system.interconnect, "device_count", 1))
    else:
        dev_cnt = 1

    init_block = TransformerBlockInitComputationTP(
        d_model=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        device_count=dev_cnt,
        data_type=cfg.data_type,
    )
    ar_block = TransformerBlockAutoRegressionTP(
        d_model=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        device_count=dev_cnt,
        data_type=cfg.data_type,
    )
    return init_block, ar_block