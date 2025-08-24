#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: compare_csv.py
Author: Julia Wen
Date: 2025-08-23
Compare two CSV audio sample files (Index,Sample) and compare two CSV spectrum files
- Handles int16, int24, and float32 CSVs
- Float32 values in [-1,1] are scaled to int16 range for comparison
- Supports --start/-s K and --limit/-n N (or a bare integer N)
- One figure with 2 subplots: overlay (top), difference (bottom)
"""
import os, sys, csv
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# ---------------- IO ----------------

def read_two_col_csv(path: str):
    """
    Expects a 2+ column CSV:
      Column 1 = axis (index or frequency)
      Column 2 = values (amplitude or magnitude)
    Returns (axis_tensor_float64, values_tensor_float64)
    """
    axis, vals = [], []
    with open(path, newline="") as fp:
        rdr = csv.reader(fp)
        for row in rdr:
            if len(row) < 2:
                continue
            try:
                a = float(row[0])  # axis
                v = float(row[1])  # values
            except Exception:
                continue
            axis.append(a)
            vals.append(v)
    if not vals:
        raise SystemExit(f"{path}: no numeric data in the first two columns.")
    A = torch.tensor(axis, dtype=torch.float64)
    V = torch.tensor(vals, dtype=torch.float64)
    # Clean NaNs
    V = torch.nan_to_num(V, nan=0.0)
    return A, V

# ---------------- WAV helpers ----------------

def looks_like_wav_index(axis: torch.Tensor) -> bool:
    """
    WAV CSVs: Column 1 is an integer sample index increasing by ~1.
    We allow small noise: >=98% integers, and >=98% of diffs ~ 1.
    """
    if axis.numel() < 3:
        return False
    # integer-ness
    ints = torch.isclose(axis, torch.round(axis), atol=1e-6)
    frac_int = ints.sum().item() / axis.numel()
    # step ~ 1
    diffs = torch.diff(axis)
    ones = torch.isclose(diffs, torch.ones_like(diffs), atol=1e-6)
    frac_one = ones.sum().item() / diffs.numel()
    return (frac_int >= 0.98) and (frac_one >= 0.98)

def normalize_wav_values(y: torch.Tensor) -> torch.Tensor:
    """
    Match typical audio CSV behavior:
      - if values are within [-1, 1], scale to int16 range for fair comparison.
    """
    maxabs = torch.max(torch.abs(y))
    if torch.isnan(maxabs) or maxabs == 0:
        return torch.zeros_like(y)
    if maxabs <= 1.0:
        return y * 32768.0
    return y

# ---------------- MAIN ----------------

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} file1.csv file2.csv [start] [limit]")
        sys.exit(1)

    f1, f2 = sys.argv[1], sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None

    a1, v1 = read_two_col_csv(f1)   # Column 1 = axis, Column 2 = values
    a2, v2 = read_two_col_csv(f2)

    wav1 = looks_like_wav_index(a1)
    wav2 = looks_like_wav_index(a2)

    if wav1 and wav2:
        # ---- WAV MODE (preserve original behavior: align by integer index) ----
        # Normalize time-domain values (float in [-1,1] -> int16 range)
        v1n = normalize_wav_values(v1)
        v2n = normalize_wav_values(v2)

        # Align by common integer indices (inner join)
        i1 = torch.round(a1).to(torch.int64)
        i2 = torch.round(a2).to(torch.int64)

        set_common = sorted(set(i1.tolist()) & set(i2.tolist()))
        if not set_common:
            raise SystemExit("No overlapping indices in WAV mode.")

        common = torch.tensor(set_common, dtype=torch.int64)

        # Build maps index -> value
        # (Using dict for exact original-style inner join semantics)
        d1 = {int(idx): float(val) for idx, val in zip(i1.tolist(), v1n.tolist())}
        d2 = {int(idx): float(val) for idx, val in zip(i2.tolist(), v2n.tolist())}

        y1_list, y2_list = [], []
        for k in common.tolist():
            if k in d1 and k in d2:
                y1_list.append(d1[k])
                y2_list.append(d2[k])

        if not y1_list or not y2_list:
            raise SystemExit("No overlapping indices after mapping in WAV mode.")

        X = common[:len(y1_list)].to(torch.float64)
        Y1 = torch.tensor(y1_list, dtype=torch.float64)
        Y2 = torch.tensor(y2_list, dtype=torch.float64)

        n_total = min(Y1.numel(), Y2.numel())
        if start < 0:
            start = 0
        if start >= n_total:
            raise SystemExit(f"Start ({start}) is >= aligned length ({n_total}).")
        end = n_total if limit is None else min(start + limit, n_total)

        Xs, Y1s, Y2s = X[start:end], Y1[start:end], Y2[start:end]
        diff = Y1s - Y2s  # keep original sign convention if needed

        print(f"\n[WAV MODE] Comparing aligned indices [{int(Xs[0].item())}:{int(Xs[-1].item())}] "
              f"(slice {start}:{end} of {n_total})")
        print(f"{os.path.basename(f1)} first 10: {Y1s[:10].tolist()}")
        print(f"{os.path.basename(f2)} first 10: {Y2s[:10].tolist()}\n")

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

        ax_top.plot(Xs.numpy(), Y1s.numpy(),
                    color="blue", linestyle="-", linewidth=3, label=os.path.basename(f1))
        ax_top.plot(Xs.numpy(), Y2s.numpy(),
                    color="orange", linestyle="--", linewidth=3, label=os.path.basename(f2))
        ax_top.set_title("Time-domain Overlay (aligned by sample index)")
        ax_top.set_xlabel("Sample index")
        ax_top.set_ylabel("Amplitude")
        ax_top.legend()

        ax_bot.plot(Xs.numpy(), (Y1s - Y2s).numpy(),
                    color="red", linewidth=2, label="Difference (file1 - file2)")
        ax_bot.axhline(0.0, linestyle="--", color="gray", linewidth=1)
        ax_bot.set_title("Difference")
        ax_bot.set_xlabel("Sample index")
        ax_bot.set_ylabel("Δ Amplitude")
        ax_bot.legend()

    else:
        # ---- SPECTRUM MODE (fixed): compare magnitudes by row; ignore axis ----
        n_total = int(min(v1.numel(), v2.numel()))
        if n_total < 2:
            raise SystemExit("Not enough rows to compare.")

        if start < 0:
            start = 0
        if start >= n_total:
            raise SystemExit(f"Start ({start}) is >= length ({n_total}).")
        end = n_total if limit is None else min(start + limit, n_total)

        Y1s = v1[start:end]
        Y2s = v2[start:end]
        Xs = torch.arange(start, end, dtype=torch.float64)
        diff = Y2s - Y1s

        print(f"\n[SPECTRUM MODE] Comparing rows [{start}:{end}] of {n_total}")
        print(f"{os.path.basename(f1)} first 10: {Y1s[:10].tolist()}")
        print(f"{os.path.basename(f2)} first 10: {Y2s[:10].tolist()}\n")

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax_top.plot(Xs.numpy(), Y1s.numpy(),
                    color="blue", linestyle="-", linewidth=3, label=os.path.basename(f1))
        ax_top.plot(Xs.numpy(), Y2s.numpy(),
                    color="orange", linestyle="--", linewidth=3, label=os.path.basename(f2))
        ax_top.set_title("Spectrum Overlay (Column 2 magnitudes)")
        ax_top.set_ylabel("Magnitude")
        ax_top.legend()

        ax_bot.plot(Xs.numpy(), diff.numpy(),
                    color="red", linewidth=2, label="Difference (file2 - file1)")
        ax_bot.axhline(0.0, linestyle="--", color="gray", linewidth=1)
        ax_bot.set_title("Difference")
        ax_bot.set_xlabel("Row (bin index in slice)")
        ax_bot.set_ylabel("Δ Magnitude")
        ax_bot.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
