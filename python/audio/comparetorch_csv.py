#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: comparetorch_csv.py
Author: Julia Wen
Date: 2025-08-22
Compare two CSV audio sample files (Index,Sample) using PyTorch.
- Aligns by Index (inner join)
- Handles int16, int24, and float32 CSVs
- Float32 values in [-1,1] are scaled to int16 range for comparison
- Supports --start/-s K and --limit/-n N (or a bare integer N)
- One figure with 2 subplots: overlay (top), difference (bottom)
"""
import sys, csv
import torch
import matplotlib.pyplot as plt

def parse_args(argv):
    if len(argv) < 3:
        print(f"Usage: {argv[0]} file1.csv file2.csv [--start K] [--limit N]")
        sys.exit(1)
    f1, f2 = argv[1], argv[2]
    start, limit = 0, None
    if "--start" in argv:
        start = int(argv[argv.index("--start")+1])
    elif "-s" in argv:
        start = int(argv[argv.index("-s")+1])
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit")+1])
    elif "-n" in argv:
        limit = int(argv[argv.index("-n")+1])
    elif len(argv) >= 4 and argv[-1].lstrip("+-").isdigit():
        limit = int(argv[-1])
    return f1, f2, start, limit

def load_csv_as_tensor(path):
    idx, val = [], []
    with open(path, newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            try:
                idx.append(int(row["Index"]))
                val.append(float(row["Sample"]))
            except:
                continue
    return torch.tensor(idx, dtype=torch.long), torch.tensor(val, dtype=torch.float64)

def normalize(y: torch.Tensor):
    if y.numel() == 0:
        return y
    # clean NaNs just in case
    y = torch.nan_to_num(y, nan=0.0)
    maxabs = torch.max(torch.abs(y))
    if maxabs <= 1.0:
        return y * 32768.0
    return y

def main(argv):
    f1, f2, start, limit = parse_args(argv)
    x1, y1 = load_csv_as_tensor(f1)
    x2, y2 = load_csv_as_tensor(f2)

    # Align by common indices
    common_idx = torch.tensor(sorted(set(x1.tolist()) & set(x2.tolist())), dtype=torch.long)
    if common_idx.numel() == 0:
        print("No overlapping indices.")
        sys.exit(2)

    # Apply start and limit
    common_idx = common_idx[start:]
    if limit is not None:
        common_idx = common_idx[:limit]

    # Map indices to values
    map1 = {int(i): float(v) for i, v in zip(x1.tolist(), y1.tolist())}
    map2 = {int(i): float(v) for i, v in zip(x2.tolist(), y2.tolist())}
    x = common_idx
    y1 = torch.tensor([map1[int(i)] for i in x.tolist()], dtype=torch.float64)
    y2 = torch.tensor([map2[int(i)] for i in x.tolist()], dtype=torch.float64)

    y1 = normalize(y1)
    y2 = normalize(y2)

    print(f"Showing {len(x)} samples from {start}")
    print(f"{f1}: min={y1.min().item():.3f}, max={y1.max().item():.3f}")
    print(f"{f2}: min={y2.min().item():.3f}, max={y2.max().item():.3f}")

    # === One figure with 2 subplots ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)

    ax1.plot(x.numpy(), y1.numpy(), label=f1, color="blue", linestyle="-")
    ax1.plot(x.numpy(), y2.numpy(), label=f2, color="orange", linestyle="--", alpha=0.8)
    ax1.legend()
    ax1.set_title("Overlay (aligned)")
    ax1.set_ylabel("Amplitude")

    ax2.plot(x.numpy(), (y1 - y2).numpy(), color="red")
    ax2.set_title("Difference (file1 - file2)")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Δ Amplitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
