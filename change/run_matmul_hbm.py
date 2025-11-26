"""
Runner: test matmul HBM 3D-stacked without transformer.

Usage (Windows PowerShell):
    python -m change.run_matmul_hbm --bs 8 --M 4096 --N 4096 --K 8192 --mode 3D_stacked

This script builds a `System` from `configs/3Dstack-template.json` and runs a
matmul operator in the chosen compile mode using `change.matmul_HBM`.
"""

import argparse

from change.dse_3D import read_architecture_template, template_to_system
from change.matmul_HBM import BatchedMatmul
from software_model.utils import data_type_dict, Tensor


def main():
    parser = argparse.ArgumentParser(description="Run HBM matmul in 3D_stacked mode without transformer")
    parser.add_argument("--bs", type=int, default=8, help="Batch size")
    parser.add_argument("--M", type=int, required=True, help="Matmul M dimension")
    parser.add_argument("--N", type=int, required=True, help="Matmul N dimension")
    parser.add_argument("--K", type=int, required=True, help="Matmul K dimension")
    parser.add_argument("--mode", choices=["3D_stacked", "heuristic-GPU"], default="3D_stacked", help="Compile mode")
    args = parser.parse_args()

    print(f"[1/4] Loading 3Dstack configuration from configs/3Dstack-template.json...")
    specs = read_architecture_template("configs/3Dstack-template.json")
    
    print(f"[2/4] Building system from architecture template...")
    system = template_to_system(specs)
    print(f"      → Device cores: {system.device.compute_module.core_count}")
    print(f"      → Memory channels: {system.device.memory_module.channel_count}")
    print(f"      → Frequency: {system.device.compute_module.clock_freq / 1e9:.2f} GHz")

    try:
        print(f"[3/4] Initializing BatchedMatmul operator (bs={args.bs}, M={args.M}, N={args.N}, K={args.K})...")
        op = BatchedMatmul(data_type=data_type_dict["fp16"])
        A = Tensor([args.bs, args.M, args.K], data_type_dict["fp16"])
        B = Tensor([args.bs, args.K, args.N], data_type_dict["fp16"])
        _ = op(A, B)
        
        print(f"[4/4] Running compile_and_simulate in mode={args.mode}...")
        latency = op.compile_and_simulate(system.device, compile_mode=args.mode)
        
        print(f"\n{'='*60}")
        print(f"Simulation Complete!")
        print(f"{'='*60}")
        print(f"Mode:            {args.mode}")
        print(f"Batch size:      {args.bs}")
        print(f"Dimensions:      M={args.M}, N={args.N}, K={args.K}")
        print(f"Simulated latency: {latency:.6f} s ({latency*1000:.3f} ms)")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n[ERROR] Matmul simulation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
