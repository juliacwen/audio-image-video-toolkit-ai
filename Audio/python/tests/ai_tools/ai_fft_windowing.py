#!/usr/bin/env python3
"""
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-12
 * Updated: 2025-09-14
 *
 * AI-assisted FFT Windowing Tool
 *
 * Manual python3 usage options:
 *   --no-plot      : disable plotting entirely
 *   --show-plot    : show plot interactively
 *   --outdir <dir> : output directory (default: test_output/)
 *
 * Requirements: Python >=3.10, PyTorch, torchaudio, matplotlib, numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

# ------------------------------
# Device detection
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Neural network
# ------------------------------
class FFTWindowSelector(nn.Module):
    """Minimal neural network predicting FFT window type based on simple waveform features."""
    def __init__(self, num_features=3, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

WINDOWS = ['rectangular', 'hann', 'hamming', 'blackman']

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(wav_path):
    waveform, sr = torchaudio.load(str(wav_path))  # cast Path to str
    waveform = waveform.mean(dim=0)
    rms = waveform.pow(2).mean().sqrt().item()
    zero_crossings = ((waveform[:-1] * waveform[1:]) < 0).sum().item()
    peak = waveform.abs().max().item()
    return torch.tensor([rms, zero_crossings / len(waveform), peak],
                        dtype=torch.float32, device=device)

# ------------------------------
# Apply FFT
# ------------------------------
def apply_fft(wav_path, model, outdir, no_plot=False, show_plot=False):
    wav_path = Path(wav_path)
    features = extract_features(wav_path)
    with torch.no_grad():
        logits = model(features)
        window_idx = logits.argmax().item()
    window_name = WINDOWS[window_idx]
    print(f"Predicted window for {wav_path.name}: {window_name}")

    waveform, sr = torchaudio.load(str(wav_path))  # cast Path to str
    waveform = waveform.mean(dim=0).numpy()
    N = len(waveform)

    if window_name == 'hann':
        win = np.hanning(N)
    elif window_name == 'hamming':
        win = np.hamming(N)
    elif window_name == 'blackman':
        win = np.blackman(N)
    else:
        win = np.ones(N)

    fft_vals = np.fft.rfft(waveform * win)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(N, 1/sr)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    out_csv = outdir / f"{wav_path.stem}_spectrum_ai.csv"
    with open(out_csv, 'w') as f:
        f.write("Frequency,Magnitude\n")
        for fr, mag in zip(freqs, fft_mag):
            f.write(f"{fr},{mag}\n")
    print(f"Spectrum saved to {out_csv}")

    # Plot
    if not no_plot:
        plt.figure()
        plt.plot(freqs, fft_mag)
        plt.title(f"FFT Spectrum ({window_name})\nFile: {wav_path.name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        if show_plot:
            plt.show()
        else:
            matplotlib.use("Agg")
            out_png = outdir / f"{wav_path.stem}_spectrum_ai.png"
            plt.savefig(out_png)
            plt.close()
            print(f"Plot saved to {out_png}")

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI-assisted FFT windowing")
    parser.add_argument("wavfile", type=str)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting entirely")
    parser.add_argument("--show-plot", action="store_true", help="Show plot interactively")
    parser.add_argument("--outdir", type=str, default="test_output", help="Output directory (default: test_output/)")
    args = parser.parse_args()

    # Environment variable for optional display (used if set)
    show_plot_env = os.environ.get("AI_FFT_SHOW_PLOT", "0") == "1"
    show_plot = args.show_plot or show_plot_env

    model = FFTWindowSelector().to(device)
    apply_fft(args.wavfile, model, outdir=args.outdir, no_plot=args.no_plot, show_plot=show_plot)

if __name__ == "__main__":
    main()

