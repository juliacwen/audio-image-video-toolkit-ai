#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: compare_csv.py
Author: Julia Wen
Date: 2025-08-19
Compare two CSV audio sample files (Index,Sample).
- Aligns by Index (inner join)
- Handles int16, int24, and float32 CSVs
- Float32 values in [-1,1] are scaled to int16 range for comparison
- Supports --start/-s K and --limit/-n N (or a bare integer N)
- One figure with 2 subplots: overlay (top), difference (bottom)
"""
import sys, csv
import numpy as np
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

def load_csv(path):
    d = {}
    with open(path, newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            try:
                d[int(row["Index"])] = float(row["Sample"])
            except:
                continue
    return d

def normalize(y):
    if y.size and np.nanmax(np.abs(y)) <= 1.0:
        return y * 32768.0
    return y

def main(argv):
    f1, f2, start, limit = parse_args(argv)
    d1, d2 = load_csv(f1), load_csv(f2)
    idx = sorted(set(d1.keys()) & set(d2.keys()))
    if not idx:
        print("No overlapping indices.")
        sys.exit(2)

    idx = idx[start:]
    if limit is not None:
        idx = idx[:limit]

    x = np.array(idx)
    y1 = normalize(np.array([d1[i] for i in idx], dtype=np.float64))
    y2 = normalize(np.array([d2[i] for i in idx], dtype=np.float64))

    print(f"Showing {len(x)} samples from {start}")
    print(f"{f1}: min={y1.min():.3f}, max={y1.max():.3f}")
    print(f"{f2}: min={y2.min():.3f}, max={y2.max():.3f}")

    # === One figure with 2 subplots ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)

    # Top: overlay
    ax1.plot(x, y1, label=f1, color="blue", linestyle="-")
    ax1.plot(x, y2, label=f2, color="orange", linestyle="--", alpha=0.8)
    ax1.legend()
    ax1.set_title("Overlay (aligned)")
    ax1.set_ylabel("Amplitude")

    # Bottom: difference
    ax2.plot(x, y1-y2, color="red")
    ax2.set_title("Difference (file1 - file2)")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Δ Amplitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
