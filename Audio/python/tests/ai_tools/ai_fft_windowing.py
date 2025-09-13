#!/usr/bin/env python3
"""
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-12

AI-assisted FFT Windowing Tool
- Extracts simple features from a WAV file
- Uses a small neural network to predict the best FFT window
- Applies FFT with that window
- Saves spectrum to CSV
- Optional plot

Requirements: Python >=3.10, PyTorch, torchaudio, matplotlib, numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------
# 1) Neural network for window selection
# ------------------------------
class FFTWindowSelector(nn.Module):
    """
    Minimal neural network predicting FFT window type
    based on simple waveform features.
    """
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
# 2) Feature extraction
# ------------------------------
def extract_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0)  # mono
    rms = waveform.pow(2).mean().sqrt().item()
    zero_crossings = ((waveform[:-1] * waveform[1:]) < 0).sum().item()
    peak = waveform.abs().max().item()
    features = torch.tensor([rms, zero_crossings / len(waveform), peak], dtype=torch.float32)
    return features

# ------------------------------
# 3) Apply FFT with predicted window
# ------------------------------
def apply_fft(wav_path, model, plot=True):
    features = extract_features(wav_path)
    with torch.no_grad():
        logits = model(features)
        window_idx = logits.argmax().item()
    window_name = WINDOWS[window_idx]
    print(f"Predicted window: {window_name}")

    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0).numpy()
    N = len(waveform)

    if window_name == 'hann':
        win = np.hanning(N)
    elif window_name == 'hamming':
        win = np.hamming(N)
    elif window_name == 'blackman':
        win = np.blackman(N)
    else:
        win = np.ones(N)  # rectangular

    fft_vals = np.fft.rfft(waveform * win)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(N, 1/sr)

    # Save spectrum CSV
    wav_path = Path(wav_path)
    out_csv = wav_path.with_name(wav_path.stem + "_spectrum_ai.csv")  # <-- FIXED
    with open(out_csv, 'w') as f:
        f.write("Frequency,Magnitude\n")
        for fr, mag in zip(freqs, fft_mag):
            f.write(f"{fr},{mag}\n")
    print(f"Spectrum saved to {out_csv}")

    # Optional plot
    if plot:
        plt.plot(freqs, fft_mag)
        plt.title(f"FFT Spectrum ({window_name})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()

# ------------------------------
# 4) Example main
# ------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI-assisted FFT windowing")
    parser.add_argument("wavfile", type=str, help="Path to WAV file")
    args = parser.parse_args()

    # Initialize model
    model = FFTWindowSelector()
    # Optionally, load pretrained weights if available:
    # model.load_state_dict(torch.load("fft_window_model.pth"))

    apply_fft(args.wavfile, model)

if __name__ == "__main__":
    main()

