#!/usr/bin/env python3
# ai_tools/ai_fft_workflow.py
"""
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-11
 
AI FFT workflow using nn_module (MLP + RNN + NN) with meaningful CSV filenames.
Generates test WAVs, spectra, trains MLP, NN, RNN and outputs predictions.
"""

import subprocess
from pathlib import Path
import numpy as np
import torch

from ai_tools import nn_module

# ----------------------------
WAV_PARAMS = [
    ("16bit", "PCM_16"),
    ("24bit", "PCM_24"),
    ("float32", "FLOAT"),
]

LABEL_MAP = {"16bit": 0, "24bit": 1, "float32": 2}

# ----------------------------
def find_wav_freq_csv() -> Path:
    """
    Locate the wav_freq_csv binary starting from the current working directory,
    then search all subdirectories. Hard-fails if not found.
    """
    start_dir = Path.cwd()
    exe = None

    # check current working directory first
    for cand in start_dir.iterdir():
        if cand.is_file() and cand.name.startswith("wav_freq_csv"):
            exe = cand
            break

    # if not found, search all subdirectories
    if exe is None:
        for cand in start_dir.rglob("wav_freq_csv*"):
            if cand.is_file():
                exe = cand
                break

    if exe is None:
        raise FileNotFoundError(
            f"Could not find wav_freq_csv binary under {start_dir} or its subdirectories. "
            f"Please build it from Audio/cpp/src/wav_freq_csv.cpp."
        )

    return exe

# ----------------------------
def run_wav_freq_csv(out_dir: Path, suffix: str, subtype: str):
    """Run wav_freq_csv to generate WAV + CSV + spectrum."""
    wav_file = out_dir / f"tone_{suffix}.wav"
    csv_file = out_dir / f"tone_{suffix}.csv"
    spectrum_file = out_dir / f"tone_{suffix}_spectrum.csv"

    exe = find_wav_freq_csv()
    cmd = [
        str(exe),
        str(wav_file),
        str(csv_file),
        str(spectrum_file),
        "--freq", "1000",
        "--sr", "8000",
        "--duration", "1.0",
        "--subtype", subtype,
    ]
    print(f"Using wav_freq_csv binary at: {exe}")
    subprocess.run(cmd, check=True)

    return wav_file, csv_file, spectrum_file

# ----------------------------
def main():
    out_dir = Path(__file__).parent / "test_output"
    out_dir.mkdir(exist_ok=True)

    # Generate WAVs + spectra
    spectrum_paths = []
    for suffix, subtype in WAV_PARAMS:
        wav_file, csv_file, spectrum_file = run_wav_freq_csv(out_dir, suffix, subtype)
        spectrum_paths.append(spectrum_file)
        print(f"WAV generated: {wav_file} ({suffix}) subtype={subtype}")

    # ----------------------------
    # Patch nn_module.load_spectra for MLP
    # ----------------------------
    original_load = nn_module.load_spectra
    def patched_load_spectra(files):
        spectra = []
        labels = []
        for f in files:
            data = np.loadtxt(f, delimiter=',', skiprows=1).flatten()
            spectra.append(data)
            for key, val in LABEL_MAP.items():
                if key in f.name:
                    labels.append(val)
                    break
            else:
                raise ValueError(f"Unknown suffix in {f.name}")
        X = np.vstack(spectra)
        y = np.array(labels)
        return X, y

    nn_module.load_spectra = patched_load_spectra
    mlp_model = nn_module.train_MLP(spectrum_paths)
    nn_module.load_spectra = original_load

    # ----------------------------
    # Patch nn_module.load_spectra_rnn for RNN
    # ----------------------------
    original_load_rnn = nn_module.load_spectra_rnn
    def patched_load_spectra_rnn(files):
        spectra = []
        labels = []
        for f in files:
            data = np.loadtxt(f, delimiter=',', skiprows=1)
            spectra.append(data)
            for key, val in LABEL_MAP.items():
                if key in f.name:
                    labels.append(val)
                    break
            else:
                raise ValueError(f"Unknown suffix in {f.name}")
        X = np.stack(spectra)  # (batch, seq_len, features)
        y = np.array(labels)
        input_size = X.shape[2]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size

    nn_module.load_spectra_rnn = patched_load_spectra_rnn
    rnn_model = nn_module.train_rnn(
        spectrum_paths,
        hidden_size=32,
        num_layers=1,
        num_classes=len(LABEL_MAP),
        epochs=10
    )
    nn_module.load_spectra_rnn = original_load_rnn

    # ----------------------------
    # Patch nn_module.load_spectra_nn for PyTorch NN
    # ----------------------------
    original_load_nn = nn_module.load_spectra_nn
    def patched_load_spectra_nn(files):
        spectra = []
        labels = []
        for f in files:
            data = np.loadtxt(f, delimiter=',', skiprows=1).flatten()
            spectra.append(data)
            for key, val in LABEL_MAP.items():
                if key in f.name:
                    labels.append(val)
                    break
            else:
                raise ValueError(f"Unknown suffix in {f.name}")
        X = np.vstack(spectra)
        y = np.array(labels)
        input_size = X.shape[1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size

    nn_module.load_spectra_nn = patched_load_spectra_nn
    nn_model = nn_module.train_nn(spectrum_paths, hidden_size=32, epochs=10)
    nn_module.load_spectra_nn = original_load_nn

    # ----------------------------
    # Predictions
    # ----------------------------
    preds = {}

    # MLP
    X, _ = patched_load_spectra(spectrum_paths)
    mlp_probs = mlp_model.predict_proba(X)
    for spec, prob in zip(spectrum_paths, mlp_probs):
        preds[(spec.name, "MLP")] = prob.tolist()

    # RNN
    X_rnn, y_rnn, _ = patched_load_spectra_rnn(spectrum_paths)
    with torch.no_grad():
        outputs = rnn_model(X_rnn)
        softmax = torch.nn.functional.softmax(outputs, dim=1).numpy()
    for spec, prob in zip(spectrum_paths, softmax):
        preds[(spec.name, "RNN")] = prob.tolist()

    # NN
    X_nn, y_nn, _ = patched_load_spectra_nn(spectrum_paths)
    with torch.no_grad():
        outputs_nn = nn_model(X_nn)
        softmax_nn = torch.nn.functional.softmax(outputs_nn, dim=1).numpy()
    for spec, prob in zip(spectrum_paths, softmax_nn):
        preds[(spec.name, "NN")] = prob.tolist()

    # ----------------------------
    # Save predictions
    # ----------------------------
    pred_file = out_dir / "predictions.txt"
    with open(pred_file, "w") as f:
        f.write("filename,model,probs\n")
        for (fname, model), prob in preds.items():
            prob_str = ",".join(f"{p:.6f}" for p in prob)
            f.write(f"{fname},{model},{prob_str}\n")

    return preds

# ----------------------------
if __name__ == "__main__":
    predictions = main()
    print("Predictions:", predictions)

