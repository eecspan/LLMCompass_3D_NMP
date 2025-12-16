import argparse
from model_configs.model import read_model_template, template_to_model_config, build_blocks_from_config
from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from hardware_model.system import system_dict


def get_system_and_params(system_name: str):
    name = system_name.lower()
    if name in ("a100", "a100_4_fp16"):
        return system_dict["A100_4_fp16"], 4, "heuristic-GPU"
    if name in ("tpuv3", "tpu", "tpuv3_8"):
        return system_dict["TPUv3_8"], 8, "heuristic-TPU"
    raise ValueError(f"Unknown system: {system_name}")


def simulate_prefill(bs, s, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode):
    model = TransformerBlockInitComputationTP(
        d_model=d_model, d_intermediate=d_intermediate, n_heads=n_heads, device_count=device_count, data_type=dtype
    )
    _ = model(Tensor([bs, s, d_model], dtype))
    if roofline:
        return model.roofline_model(system)
    else:
        return model.compile_and_simulate(system, compile_mode)


def simulate_decode_step(bs, seq_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode):
    model = TransformerBlockAutoRegressionTP(
        d_model=d_model, d_intermediate=d_intermediate, n_heads=n_heads, device_count=device_count, data_type=dtype
    )
    _ = model(Tensor([bs, 1, d_model], dtype), seq_len)
    if roofline:
        return model.roofline_model(system)
    else:
        return model.compile_and_simulate(system, compile_mode)


def decode_last(bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode):
    # 与参考脚本一致：只在 seq_len = s + out_len 下模拟一次单步延迟
    if out_len <= 0:
        return 0.0
    last_step = simulate_decode_step(bs, s + out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode)
    return last_step * out_len


def decode_exact(bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode):
    # 逐 token 累加，最精确但开销大
    total = 0.0
    for t in range(out_len):
        total += simulate_decode_step(bs, s + t, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode)
    return total


def decode_fit2(bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode):
    # 线性两点拟合：lat(seq) ≈ c0 + c1 * seq
    # L0 = lat(s), L1 = lat(s+out_len) → c1 = (L1 - L0)/out_len, c0 = L0 - c1*s
    # sum_{i=0..out_len-1} lat(s+i) = out_len*c0 + c1*(out_len*s + out_len*(out_len-1)/2)
    if out_len <= 0:
        return 0.0
    L0 = simulate_decode_step(bs, s, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode)
    L1 = simulate_decode_step(bs, s + out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline, compile_mode)
    c1 = (L1 - L0) / max(out_len, 1)
    c0 = L0 - c1 * s
    total = out_len * c0 + c1 * (out_len * s + out_len * (out_len - 1) / 2.0)
    return total


def main():
    parser = argparse.ArgumentParser()
    # 新增模型配置参数
    parser.add_argument("--model_config", type=str, required=False, help="path to model json")
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--input_len", type=int, required=True)
    parser.add_argument("--output_len", type=int, required=True)
    parser.add_argument("--system_config", type=str, default="configs/GA100.json")
    parser.add_argument("--roofline", action="store_true")
    parser.add_argument("--decode_mode", type=str, default="last", choices=["last","exact","fit2"])
    args = parser.parse_args()
    # 1) 加载系统
    arch_specs = read_architecture_template(args.system_config)
    system = template_to_system(arch_specs)
    # 2) 加载模型配置（若提供）
    if args.model_config:
        model_specs = read_model_template(args.model_config)
        cfg = template_to_model_config(model_specs, default_device_count=system.interconnect.device_count)
        d_model = cfg.hidden_size
        d_intermediate = cfg.intermediate_size
        n_heads = cfg.num_attention_heads
        layers = cfg.num_hidden_layers
        dtype = cfg.data_type
        device_count = cfg.device_count or system.interconnect.device_count
    else:
        raise ValueError("Model configuration is required")
    # 3) 构建算子
    from software_model.utils import Tensor
    from software_model.transformer import TransformerBlockInitComputationTP, TransformerBlockAutoRegressionTP

    prefill_block = TransformerBlockInitComputationTP(d_model, d_intermediate, n_heads, device_count, dtype)
    decode_block  = TransformerBlockAutoRegressionTP(d_model, d_intermediate, n_heads, device_count, dtype)

    # 4) 计算 prefill
    _ = prefill_block(Tensor([args.bs, args.input_len, d_model], dtype))
    prefill_latency = (
        prefill_block.roofline_model(system) if args.roofline
        else prefill_block.compile_and_simulate(system, "heuristic-GPU")
    )

    # 5) 计算 decode（支持 last/exact/fit2）
    def step_latency(seq_len):
        _ = decode_block(Tensor([args.bs, 1, d_model], dtype), seq_len)
        return (
            decode_block.roofline_model(system) if args.roofline
            else decode_block.compile_and_simulate(system, "heuristic-GPU")
        )

    out_len = args.output_len
    s = args.input_len
    if out_len <= 0:
        decode_total = 0.0
    elif args.decode_mode == "last":
        L_last = step_latency(s + out_len)
        decode_total = L_last * out_len
    elif args.decode_mode == "exact":
        decode_total = sum(step_latency(s + t) for t in range(out_len))
    else:
        L0 = step_latency(s)
        L1 = step_latency(s + out_len)
        c1 = (L1 - L0) / max(out_len, 1)
        c0 = L0 - c1 * s
        decode_total = out_len * c0 + c1 * (out_len * s + out_len * (out_len - 1) / 2.0)
    # 6) 按层数放大
    print(f"prefill_latency: {prefill_latency * layers}")
    print(f"decode_latency:  {decode_total * layers}")


if __name__ == "__main__":
    main()