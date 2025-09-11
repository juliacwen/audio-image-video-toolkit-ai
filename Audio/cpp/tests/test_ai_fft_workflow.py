#!/usr/bin/env python3
# test_ai_fft_workflow.py
"""
    -test_ai_fft_workflow.py
     *Author: Julia Wen wendigilane@gmail.com
     *Date: 2025-09-10

    End-to-end Wav Spectra test:
    - Generates 3 WAVs
    - Converts WAV → CSV + spectrum
    - Runs NN on all spectra:
        * MLP = Multi-Layer Perceptron (feedforward neural network)
        * RNN = Recurrent Neural Network (LSTM-based for sequential data)
    - Validates predictions
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch
import pytest

from ai_tools import nn_module

# ----------------------------
# Map file suffix to integer label
# ----------------------------
LABEL_MAP = {"16bit": 0, "24bit": 1, "float32": 2}

# ----------------------------
# Parameterized test for models + files
# ----------------------------
@pytest.mark.parametrize("model_type", ["MLP", "RNN"])
@pytest.mark.parametrize("suffix", ["16bit", "24bit", "float32"])
def test_model_training_per_file(suffix, model_type):
    out_dir = Path(__file__).parent / "ai_tools" / "test_output"
    spectrum_file = out_dir / f"tone_{suffix}_spectrum.csv"

    # Patch label mapping for this test
    def patched_label_from_file(f):
        return LABEL_MAP[suffix]

    # ----------------------------
    # MLP
    # ----------------------------
    if model_type == "MLP":
        original_load = nn_module.load_spectra
        def patched_load_spectra(files):
            data = np.loadtxt(files[0], delimiter=',', skiprows=1).flatten()
            X = np.array([data])
            y = np.array([patched_label_from_file(files[0])])
            return X, y
        nn_module.load_spectra = patched_load_spectra

        model = nn_module.train_nn([spectrum_file])
        assert model is not None
        print(f"MLP model trained successfully on {suffix} spectrum.")
        nn_module.load_spectra = original_load

    # ----------------------------
    # RNN
    # ----------------------------
    elif model_type == "RNN":
        original_load_rnn = nn_module.load_spectra_rnn
        def patched_load_spectra_rnn(files):
            data = np.loadtxt(files[0], delimiter=',', skiprows=1)
            X = data[np.newaxis, :, :]  # (1, seq_len, features)
            y = np.array([patched_label_from_file(files[0])])
            input_size = X.shape[2]
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size
        nn_module.load_spectra_rnn = patched_load_spectra_rnn

        model = nn_module.train_rnn([spectrum_file], hidden_size=16, num_layers=1, num_classes=len(LABEL_MAP), epochs=5)
        assert isinstance(model, nn_module.SpectrumRNN)
        print(f"RNN model trained successfully on {suffix} spectrum.")

        X, y, _ = nn_module.load_spectra_rnn([spectrum_file])
        with torch.no_grad():
            outputs = model(X)
        assert outputs.shape[0] == 1
        assert outputs.shape[1] == len(LABEL_MAP)
        print(f"RNN forward pass successful, output shape verified for {suffix}.")

        nn_module.load_spectra_rnn = original_load_rnn

# ----------------------------
# Existing AI FFT workflow validation
# ----------------------------
def test_ai_fft_predictions():
    script = Path(__file__).parent / "ai_tools" / "ai_fft_workflow.py"

    # Load module and run main()
    spec = importlib.util.spec_from_file_location("ai_fft_workflow", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preds_direct = module.main()  # returns dict {filename: array [class0,class1]}

    # Load predictions.txt
    out_dir = Path(__file__).parent / "ai_tools" / "test_output"
    pred_file = out_dir / "predictions.txt"
    assert pred_file.exists(), "predictions.txt was not created"

    # === Load from file ===
    file_names = np.loadtxt(pred_file, delimiter=",", skiprows=1, usecols=0, dtype=str)
    preds_file = np.loadtxt(pred_file, delimiter=",", skiprows=1, usecols=(1,2))

    # === Align dict predictions with file order ===
    preds_from_dict = np.array([preds_direct[f] for f in file_names])

    # === Basic assertions ===
    assert isinstance(preds_direct, dict)
    assert len(preds_direct) == 3
    for fname, prob in preds_direct.items():
        prob_arr = np.array(prob)
        assert np.all((prob_arr >= 0) & (prob_arr <= 1))

    # === Compare predictions dict vs file ===
    assert np.allclose(preds_from_dict, preds_file, atol=1e-8), "Predictions mismatch between dict and file"

    # === Multi-class softmax, check row sums ~1 ===
    preds_array = np.vstack([np.array(preds_direct[f]) for f in file_names])
    for row in preds_array:
        assert np.isclose(np.sum(row), 1.0, atol=1e-3), f"Row does not sum to 1: {row}"

    # === Ensure WAV, CSV, and spectrum files exist ===
    for suffix in ["16bit", "24bit", "float32"]:
        wav_file = out_dir / f"tone_{suffix}.wav"
        csv_file = out_dir / f"tone_{suffix}.csv"
        spectrum_file = out_dir / f"tone_{suffix}_spectrum.csv"
        assert wav_file.exists(), f"Missing WAV file: {wav_file}"
        assert csv_file.exists(), f"Missing CSV file: {csv_file}"
        assert spectrum_file.exists(), f"Missing spectrum file: {spectrum_file}"

