#!/usr/bin/env python3
"""
test_ai_fft_workflow.py

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-11

 -nn_module unit tests (MLP, NN, RNN) and end-to-end workflow test.
"""

import importlib.util
from pathlib import Path
import numpy as np
import torch
import pytest

from ai_tools import nn_module

# ----------------------------
# Label map for files
# ----------------------------
LABEL_MAP = {"16bit": 0, "24bit": 1, "float32": 2}

# ----------------------------
# Existing parameterized tests for nn_module
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

        model = nn_module.train_MLP([spectrum_file])
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
# New PyTorch NN tests
# ----------------------------
@pytest.mark.parametrize("suffix", ["16bit", "24bit", "float32"])
def test_model_training_per_file_nn(suffix):
    """
    Test new PyTorch feedforward NN (SpectrumNN) per spectrum file.
    """
    out_dir = Path(__file__).parent / "ai_tools" / "test_output"
    spectrum_file = out_dir / f"tone_{suffix}_spectrum.csv"

    original_load = nn_module.load_spectra_nn
    def patched_load_spectra_nn(files):
        data = np.loadtxt(files[0], delimiter=',', skiprows=1).flatten()
        X = np.array([data])
        y = np.array([LABEL_MAP[suffix]])
        input_size = X.shape[1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size
    nn_module.load_spectra_nn = patched_load_spectra_nn

    model = nn_module.train_nn([spectrum_file], hidden_size=16, epochs=5)
    assert isinstance(model, nn_module.SpectrumNN)
    print(f"PyTorch NN model trained successfully on {suffix} spectrum.")

    X, y, _ = nn_module.load_spectra_nn([spectrum_file])
    with torch.no_grad():
        outputs = model(X)
    assert outputs.shape[0] == 1
    assert outputs.shape[1] == len(LABEL_MAP)
    print(f"SpectrumNN forward pass successful, output shape verified for {suffix}.")

    nn_module.load_spectra_nn = original_load

# ----------------------------
# End-to-end workflow loader
# ----------------------------
def load_workflow_module():
    script = Path(__file__).parent / "ai_tools" / "ai_fft_workflow.py"
    spec = importlib.util.spec_from_file_location("ai_fft_workflow", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ----------------------------
# Parameterized end-to-end workflow test
# ----------------------------
@pytest.mark.parametrize("suffix", ["16bit", "24bit", "float32"])
@pytest.mark.parametrize("model_type", ["MLP", "RNN", "NN"])
def test_ai_fft_predictions_param(suffix, model_type):
    """End-to-end workflow: generate WAVs, spectra, train nn_module model, validate predictions."""
    module = load_workflow_module()
    preds = module.main()  # dict {(filename, model): probs}

    out_dir = Path(__file__).parent / "ai_tools" / "test_output"
    pred_file = out_dir / "predictions.txt"
    assert pred_file.exists(), "predictions.txt was not created"

    fname = f"tone_{suffix}_spectrum.csv"
    key = (fname, model_type)
    assert key in preds, f"Prediction missing for {fname} {model_type}"

    arr = np.array(preds[key])
    # Must match number of classes
    assert arr.shape[0] == len(LABEL_MAP)
    # Probabilities in [0,1]
    assert np.all((arr >= 0) & (arr <= 1))
    # Sum of probabilities ~1
    assert np.isclose(arr.sum(), 1.0, atol=1e-3)

    # Check WAV, CSV, spectrum existence
    wav_file = out_dir / f"tone_{suffix}.wav"
    csv_file = out_dir / f"tone_{suffix}.csv"
    spec_file = out_dir / f"tone_{suffix}_spectrum.csv"
    assert wav_file.exists()
    assert csv_file.exists()
    assert spec_file.exists()

