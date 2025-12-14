import subprocess
import math
import os
import pandas as pd
import argparse
from collections import defaultdict

# Constants for HBM geometry and granularity
channel_per_nmp = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32  # Byte
data_size = 2
n_mac = int(prefetch_size / data_size)

# HBM Granularity Sizes
HBM_GS = {
    'col': prefetch_size,
    'row': n_col * prefetch_size,
    'ba': n_row * n_col * prefetch_size,
    'bg': n_bank * n_row * n_col * prefetch_size,
    'rank': n_bg * n_bank * n_row * n_col * prefetch_size,
    'pch': n_rank * n_bg * n_bank * n_row * n_col * prefetch_size,
    'ch': n_pch * n_rank * n_bg * n_bank * n_row * n_col * prefetch_size
}

cmd_fc_wrgb = []
cmd_fc_mac = []
cmd_fc_mvsb = []


def apply_config(geom: dict | None = None):
    """
    根据 config 更新几何参数，并重算 HBM_GS。
    仅更新提供的键，未提供的保持默认。
    """
    global channel_per_nmp, n_pch, n_rank, n_bank, n_bg, n_row, n_col, prefetch_size, data_size, HBM_GS
    if geom:
        for k in ["channel_per_nmp","n_pch","n_rank","n_bank","n_bg","n_row","n_col","prefetch_size","data_size"]:
            if k in geom:
                globals()[k] = geom[k]

    HBM_GS = {
        'col': prefetch_size,
        'row': n_col * prefetch_size,
        'ba': n_row * n_col * prefetch_size,
        'bg': n_bank * n_row * n_col * prefetch_size,
        'rank': n_bg * n_bank * n_row * n_col * prefetch_size,
        'pch': n_rank * n_bg * n_bank * n_row * n_col * prefetch_size,
        'ch': n_pch * n_rank * n_bg * n_bank * n_row * n_col * prefetch_size
    }


def cmd_list_reset():
    global cmd_fc_wrgb, cmd_fc_mac, cmd_fc_mvsb
    cmd_fc_wrgb = []
    cmd_fc_mac = []
    cmd_fc_mvsb = []

def mapping_func(addr, K, ch_number):
    ch_bit = int(math.log2(ch_number))
    # Compute the number of bits used to encode column index
    # col1 = (addr >> 4) & 0b111
    col = (addr >> 5) & 0b11111  # [4:0]
    k_bits = 0
    # k_bits = int(math.log2(2 * K / 1024))  # K is the number of columns in the matrix
    # # print(k_bits)
    # row_bits = (1<<k_bits)-1 if k_bits > 0 else 0

    # row1 = (addr >> 10) & row_bits  # Shift right by 5 to remove column bits
    # col2 = (addr >> (7+k_bits)) & 0b111
    row = (addr >> (6+4+k_bits+ch_bit+1+1+2+2))
    bank       = (addr >> (6+4+k_bits+ch_bit+1+1+2)) & 0b11      
    bankgroup  = (addr >> (6+4+k_bits+ch_bit+1+1)) & 0b11      
    rank       = (addr >> (6+4+k_bits+ch_bit+1)) & 0b1       
    pch        = (addr >> (6+4+k_bits+ch_bit)) & 0b1       
    ch         = (addr >> (6+4+k_bits))  & (((1 << ch_bit) - 1)) if ch_bit > 0 else 0   

    # row = (row2 << k_bits) + row1 if k_bits > 0 else row2
    # col = (col2 << 3) + col1

    return {
        "col": col,
        "row": row,
        "bank": bank,
        "bankgroup": bankgroup,
        "rank": rank,
        "pch": pch,
        "ch": ch
    }

def generate_trace_per_channel(address_list):
    """
    每个 channel 内按 (row, col) 去重，只按 row 排序。
    返回: dict[ch] -> list[addr]
    """
    ch_groups = defaultdict(list)
    seen_rc_per_ch = defaultdict(set)  # 每个 channel 已见 (row, col)

    for addr in address_list:
        ch = addr["ch"]
        rc = (addr["row"], addr["col"])
        if rc in seen_rc_per_ch[ch]:
            continue
        seen_rc_per_ch[ch].add(rc)
        ch_groups[ch].append(addr.copy())  # 拷贝避免后续修改污染

    # 每个 channel 内按 row 排序
    for ch in ch_groups:
        ch_groups[ch].sort(key=lambda x: x["row"])

        # 自检
        keys = [(a["row"], a["col"]) for a in ch_groups[ch]]
        assert len(keys) == len(set(keys)), f"Channel {ch}: duplicated (row,col) after dedup -> {keys}"

    return ch_groups

def FC_bank_level(N, K, data_size, addr_offset, itr, valid_channel=None):
    if valid_channel is None:
        valid_channel = channel_per_nmp
    print("valid channel: ", valid_channel)
    # batch_list = []
    col_set = set()
    row_set = set()
    addr_list = []
    new_mac = prefetch_size//data_size
    for n_idx in range(N):
        for k_idx in range(math.ceil(K / new_mac)):
            idx = k_idx*new_mac*data_size + n_idx * K *data_size
            addr_idx = mapping_func(idx, K, valid_channel)
            if addr_idx['col'] not in col_set:
                col_set.add(addr_idx['col'])
                row_set.add(addr_idx['row'])
            addr_list.append(addr_idx)
    # batch_list.append(addr_list)
    print(col_set)
    print(row_set)
    return addr_list

def interleave_and_generate_trace(ch_groups, inst_str):
    """
    ch_groups: dict[ch] -> list of address dicts
    HBM_GS: dict，存储 col/row/bank/pch/ch/bankgroup 对应的 stride
    返回: list[str]，每个元素是一行 trace
    """
    idx = {ch: 0 for ch in ch_groups}  # 每个 channel 当前取到的位置
    total_len = sum(len(lst) for lst in ch_groups.values())
    print(total_len)
    final_trace = []

    print(sorted(ch_groups.keys()))

    while len(final_trace) < total_len:
        for ch in sorted(ch_groups.keys()):  # 轮询 channel
            if idx[ch] < len(ch_groups[ch]):
                # 拿到当前地址（拷贝避免意外修改原始数据）
                addr_idx = ch_groups[ch][idx[ch]]
                # print(addr_idx)

                # 计算物理地址，不修改 addr_idx
                addr_val = (
                    addr_idx['col']       * HBM_GS['col'] +
                    addr_idx['row']       * HBM_GS['row'] +
                    addr_idx['bank']      * HBM_GS['ba'] +
                    addr_idx['pch']       * HBM_GS['pch'] +
                    addr_idx['ch']        * HBM_GS['ch'] +
                    addr_idx['bankgroup'] * HBM_GS['bg'] +
                    addr_idx['rank'] * HBM_GS['rank']
                ) 
                final_trace.append(f"{inst_str} 0x{addr_val:08x}\n")
                idx[ch] += 1  # 前进到下一个地址
    return final_trace


def run_fc(inst_str, N, K, data_size, trace_file_name):
    cmd_list_reset()
    base_addr = 0
    itr = 0
    merged_ch_groups = defaultdict(list)
    addr_list = FC_bank_level(N, K, data_size, base_addr, itr)
    # print(addr_list)
    print(len(addr_list))
    # for addr_list in batch_list:
    ch_groups = generate_trace_per_channel(addr_list)
    final_trace = interleave_and_generate_trace(ch_groups, inst_str)
    with open(trace_file_name, 'w') as trace_file:
        for cmd in final_trace:
            trace_file.write(cmd)
    trace_file.close()


def make_yaml_file(yaml_file,file_name,channel,mapping):
        trace_path = file_name
        line = ""
        line += "Frontend:\n"
        line += "  impl: LoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: GenericDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: HBM3\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += "      channel: {}\n".format(channel)
        line += "    timing:\n"
        line += "      preset: HBM3_5.2Gbps\n"
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: HBM3\n"
        line += "    Scheduler:\n"
        line += "      impl: FRFCFS\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "      - ControllerPlugin:\n"
        line += "          impl: HBM3TraceRecorder\n"
        line += "          path: /home/panyudong/work/3d-monitor/attacc_simulator/ramulator2/log/load/load_cmd.log\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: HBM3-Custom\n"
        with open(yaml_file, 'w') as f:
            f.write(line)

def load_store_cycle_func(
    inst_type: int = 0, # 0 for load, 1 for store
    fcN: int = 512,
    fcK: int = 512,
    data_size: int = 2,
    mapping: str = "ChRaBaRoCo",
    ramulator_path: str = "/home/panyudong/work/3d-monitor/attacc_simulator/ramulator2/ramulator2",
    clean_temp: bool = False,
) -> int:
    if inst_type == 0:
        inst_str = "LD"
        name = "load-data"
        output = "load_bank_level.trace"
    elif inst_type == 1:
        inst_str = "ST"
        name = "store-data"
        output = "store_bank_level.trace"
    else:
        raise ValueError(f"Invalid inst_type: {inst_type}")
    
    # === Step 1: Generate trace using your actual FC trace generator ===
    print(f"[Trace] Generating trace with run_fc(): {output}")
    run_fc(inst_str, fcN, fcK, data_size, output)

    # === Step 2: Generate YAML config ===
    channel = channel_per_nmp  # fixed
    yaml_name = f"{name}_ch{channel}_{mapping}.yaml"
    make_yaml_file(yaml_name, output, channel, mapping)

    # === Step 3: Run Ramulator2 ===
    run_cmd = f"{ramulator_path} -f {yaml_name}"
    print(f"[Sim] Running: {run_cmd}")
    try:
        result = subprocess.run(run_cmd, shell=True, check=True, capture_output=True, text=True)
        output_lines = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ramulator failed: {e}")
        return -1  # or raise exception

    # === Step 4: Parse memory_system_cycles ===
    cycle = 0
    for line in output_lines:
        if "memory_system_cycles" in line:
            try:
                cycle += int(line.strip().split()[-1])
            except Exception:
                pass

    print(f"[Result] Model={name}, fcN={fcN}, fcK={fcK}, cycle={cycle}")

    # === Step 5: Clean up temp files if needed ===
    if clean_temp:
        try:
            os.remove(yaml_name)
            os.remove(output)
        except Exception as e:
            print(f"[WARN] Failed to delete temp files: {e}")
    return cycle

if __name__ == "__main__":
    cycle = load_store_cycle_func(1)
    print(cycle)