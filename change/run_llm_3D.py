"""
Run LLM latency simulation on 3Dstack architecture (prefill + decode).
Usage:
    python -m change.run_llm_3D \
        --model_config configs/models/llama_3.1_8b.json \
        --system_config configs/3Dstack-template.json \
        --bs 8 --input_len 2048 --output_len 256 \
        --decode_mode last \
        [--roofline]
"""

import argparse

from model_configs.model import read_model_template, template_to_model_config
from change.dse_3D import read_architecture_template, template_to_system
from change.transformer_3D import (
    TransformerBlockInitComputationTP3D,
    TransformerBlockAutoRegressionTP3D,
)
from software_model.utils import Tensor

def simulate_prefill(bs, s, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline):
    model = TransformerBlockInitComputationTP3D(
        d_model=d_model,
        d_intermediate=d_intermediate,
        n_heads=n_heads,
        device_count=device_count,
        data_type=dtype,
    )
    _ = model(Tensor([bs, s, d_model], dtype))
    return model.roofline_model(system) if roofline else model.compile_and_simulate(system, "3D_stacked")

def simulate_decode_step(bs, seq_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline):
    model = TransformerBlockAutoRegressionTP3D(
        d_model=d_model,
        d_intermediate=d_intermediate,
        n_heads=n_heads,
        device_count=device_count,
        data_type=dtype,
    )
    _ = model(Tensor([bs, 1, d_model], dtype), seq_len)
    return model.roofline_model(system) if roofline else model.compile_and_simulate(system, "3D_stacked")

def decode_last(bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline):
    if out_len <= 0:
        return 0.0
    last = simulate_decode_step(bs, s + out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline)
    return last * out_len

def decode_exact(bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline):
    total = 0.0
    for t in range(out_len):
        total += simulate_decode_step(bs, s + t, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline)
    return total

def decode_fit2(bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline):
    if out_len <= 0:
        return 0.0
    L0 = simulate_decode_step(bs, s, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline)
    L1 = simulate_decode_step(bs, s + out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, roofline)
    c1 = (L1 - L0) / max(out_len, 1)
    c0 = L0 - c1 * s
    return out_len * c0 + c1 * (out_len * s + out_len * (out_len - 1) / 2.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--system_config", type=str, default="configs/3Dstack-template.json")
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--input_len", type=int, required=True)
    parser.add_argument("--output_len", type=int, required=True)
    parser.add_argument("--roofline", action="store_true")
    parser.add_argument("--decode_mode", type=str, default="last", choices=["last", "exact", "fit2"])
    args = parser.parse_args()

    # 1) 系统
    arch_specs = read_architecture_template(args.system_config)
    system = template_to_system(arch_specs)

    # 2) 模型配置
    model_specs = read_model_template(args.model_config)
    cfg = template_to_model_config(model_specs, default_device_count=system.interconnect.device_count)
    d_model = cfg.hidden_size
    d_intermediate = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    layers = cfg.num_hidden_layers
    dtype = cfg.data_type
    device_count = cfg.device_count or system.interconnect.device_count

    # 3) prefill
    prefill_latency = simulate_prefill(args.bs, args.input_len, d_model, d_intermediate, n_heads, device_count, dtype, system, args.roofline)

    # 4) decode
    s = args.input_len
    out_len = args.output_len
    if args.decode_mode == "last":
        decode_total = decode_last(args.bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, args.roofline)
    elif args.decode_mode == "exact":
        decode_total = decode_exact(args.bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, args.roofline)
    else:
        decode_total = decode_fit2(args.bs, s, out_len, d_model, d_intermediate, n_heads, device_count, dtype, system, args.roofline)

    # 5) 按层数放大
    print(f"prefill_latency: {prefill_latency * layers}")
    print(f"decode_latency:  {decode_total * layers}")

if __name__ == "__main__":
    main()