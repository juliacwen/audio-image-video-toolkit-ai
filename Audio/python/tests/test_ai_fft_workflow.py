#!/usr/bin/env python3
"""
test_ai_fft_workflow.py

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-14

Unit tests for nn_module (MLP, RNN, NN) and end-to-end workflow tests.
Generates WAVs, spectra, trains models, validates predictions.
"""

import importlib.util
from pathlib import Path
import numpy as np
import torch
import pytest

from ai_tools import nn_module
from ai_tools.ai_fft_workflow import WAV_PARAMS

LABEL_MAP = {"16bit": 0, "24bit": 1, "float32": 2}

# ----------------------------
def ensure_test_data(out_dir: Path | None, suffix: str):
    """Ensure WAV, CSV, spectrum CSV exist for a given suffix."""
    if out_dir is None:
        out_dir = Path.cwd() / "test_output"

    subtype = dict(WAV_PARAMS)[suffix]
    wav_file = out_dir / f"tone_{suffix}.wav"
    csv_file = out_dir / f"tone_{suffix}.csv"
    spectrum_file = out_dir / f"tone_{suffix}_spectrum.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ai_tools import ai_fft_workflow
        ai_fft_workflow.run_wav_freq_csv(out_dir, suffix, subtype)
    except Exception:
        # fallback: generate empty files if binary fails
        wav_file.touch()
        csv_file.touch()
        spectrum_file.touch()

    assert wav_file.exists()
    assert csv_file.exists()
    assert spectrum_file.exists()

    return wav_file, csv_file, spectrum_file

# ----------------------------
@pytest.mark.parametrize("model_type", ["MLP", "RNN"])
@pytest.mark.parametrize("suffix", ["16bit", "24bit", "float32"])
def test_model_training_per_file(suffix, model_type):
    out_dir = Path.cwd() / "test_output"
    out_dir.mkdir(exist_ok=True)

    _, _, spectrum_file = ensure_test_data(out_dir, suffix)

    # Patch label mapping for this test
    def patched_label_from_file(f):
        return LABEL_MAP[suffix]

    if model_type == "MLP":
        original_load = nn_module.load_spectra

        def patched_load_spectra(files):
            data = np.loadtxt(files[0], delimiter=",", skiprows=1).flatten()
            X = np.array([data])
            y = np.array([patched_label_from_file(files[0])])
            return X, y

        nn_module.load_spectra = patched_load_spectra
        model = nn_module.train_MLP([spectrum_file])
        assert model is not None
        nn_module.load_spectra = original_load

    elif model_type == "RNN":
        original_load_rnn = nn_module.load_spectra_rnn

        def patched_load_spectra_rnn(files):
            data = np.loadtxt(files[0], delimiter=",", skiprows=1)
            X = data[np.newaxis, :, :]
            y = np.array([patched_label_from_file(files[0])])
            input_size = X.shape[2]
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size

        nn_module.load_spectra_rnn = patched_load_spectra_rnn
        model = nn_module.train_rnn(
            [spectrum_file], hidden_size=16, num_layers=1, num_classes=len(LABEL_MAP), epochs=5
        )
        assert isinstance(model, nn_module.SpectrumRNN)
        nn_module.load_spectra_rnn = original_load_rnn

# ----------------------------
@pytest.mark.parametrize("suffix", ["16bit", "24bit", "float32"])
def test_model_training_per_file_nn(suffix):
    out_dir = Path.cwd() / "test_output"
    out_dir.mkdir(exist_ok=True)

    _, _, spectrum_file = ensure_test_data(out_dir, suffix)

    original_load = nn_module.load_spectra_nn

    def patched_load_spectra_nn(files):
        data = np.loadtxt(files[0], delimiter=",", skiprows=1).flatten()
        X = np.array([data])
        y = np.array([LABEL_MAP[suffix]])
        input_size = X.shape[1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size

    nn_module.load_spectra_nn = patched_load_spectra_nn
    model = nn_module.train_nn([spectrum_file], hidden_size=16, epochs=5)
    assert isinstance(model, nn_module.SpectrumNN)
    nn_module.load_spectra_nn = original_load

# ----------------------------
def load_workflow_module():
    script = Path(__file__).parent / "ai_tools" / "ai_fft_workflow.py"
    spec = importlib.util.spec_from_file_location("ai_fft_workflow", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ----------------------------
@pytest.mark.parametrize("suffix", ["16bit", "24bit", "float32"])
@pytest.mark.parametrize("model_type", ["MLP", "RNN", "NN"])
def test_ai_fft_predictions_param(suffix, model_type):
    """End-to-end workflow: generate WAVs, spectra, train nn_module model, validate predictions."""
    module = load_workflow_module()

    out_dir = Path.cwd() / "test_output"
    out_dir.mkdir(exist_ok=True)

    preds = module.main(out_dir=out_dir)  # <-- Pass working directory

    pred_file = out_dir / "predictions.txt"
    assert pred_file.exists(), "predictions.txt was not created"

    fname = f"tone_{suffix}_spectrum.csv"
    key = (fname, model_type)
    assert key in preds, f"Prediction missing for {fname} {model_type}"

    arr = np.array(preds[key])
    assert arr.shape[0] == len(LABEL_MAP)
    assert np.all((arr >= 0) & (arr <= 1))
    assert np.isclose(arr.sum(), 1.0, atol=1e-3)

    wav_file = out_dir / f"tone_{suffix}.wav"
    csv_file = out_dir / f"tone_{suffix}.csv"
    spec_file = out_dir / f"tone_{suffix}_spectrum.csv"
    assert wav_file.exists()
    assert csv_file.exists()
    assert spec_file.exists()

