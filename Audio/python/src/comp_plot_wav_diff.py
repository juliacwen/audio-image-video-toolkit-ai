#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: compare_plot_wav_diff.py
Author: Julia Wen
Date: 2025-08-19
Description: 
    This script compares two wav files and plot
"""

import torch
import torchaudio
import argparse
import matplotlib.pyplot as plt

def compare_and_plot(file1, file2, zoom=None, show=True, save=None):
    # Load both files
    wav1, sr1 = torchaudio.load(file1)
    wav2, sr2 = torchaudio.load(file2)

    print(f"{file1}: shape={wav1.shape}, sample_rate={sr1}")
    print(f"{file2}: shape={wav2.shape}, sample_rate={sr2}")

    if sr1 != sr2:
        raise ValueError(f"Sample rates differ: {sr1} vs {sr2}")

    # Align lengths
    n = min(wav1.shape[1], wav2.shape[1])
    wav1, wav2 = wav1[:, :n], wav2[:, :n]

    # Metrics
    mse = torch.mean((wav1 - wav2) ** 2).item()
    corr = torch.corrcoef(torch.stack([wav1.flatten(), wav2.flatten()]))[0, 1].item()

    print("\nComparison Metrics:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  Correlation:        {corr:.4f}")

    # Plot
    time = torch.arange(n) / sr1
    plt.figure(figsize=(12, 8))

    # Top: both signals
    plt.subplot(2, 1, 1)
    plt.plot(time, wav1[0], label=file1, alpha=0.7, color="blue", linewidth=1)
    plt.plot(time, wav2[0], label=file2, alpha=0.7, color="orange", linestyle="--", linewidth=1)
    plt.title("Waveforms (Overlay)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    if zoom:
        plt.xlim(0, zoom)  # zoom in on first N seconds

    # Bottom: difference
    plt.subplot(2, 1, 2)
    plt.plot(time, (wav1 - wav2)[0], color="red", alpha=0.8)
    plt.title("Difference (wav1 - wav2)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude Difference")
    plt.grid(True)

    if zoom:
        plt.xlim(0, zoom)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
        print(f"Plot saved to {save}")
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare and plot two WAV files")
    parser.add_argument("file1", help="First WAV file")
    parser.add_argument("file2", help="Second WAV file")
    parser.add_argument("--save", help="Save plot to image file")
    parser.add_argument("--zoom", type=float, help="Zoom in on first N seconds")
    args = parser.parse_args()

    compare_and_plot(args.file1, args.file2, zoom=args.zoom, save=args.save)