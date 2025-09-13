#!/usr/bin/env python3
"""
ai_tools/nn_module.py - Spectra NN module with optional RNN (auto-detect sequence length)

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-11

Example usage:

# ----------------------------
# Train scikit-learn MLP (existing workflow)
# ----------------------------
from ai_tools import nn_module
mlp_model = nn_module.train_MLP(["tone_16bit_spectrum.csv", "tone_24bit_spectrum.csv"])

# ----------------------------
# Train PyTorch feedforward NN (new)
# ----------------------------
from ai_tools import nn_module
nn_model = nn_module.train_nn(["tone_16bit_spectrum.csv", "tone_24bit_spectrum.csv"], hidden_size=64, epochs=10)

# ----------------------------
# Train RNN (existing workflow)
# ----------------------------
from ai_tools import nn_module
rnn_model = nn_module.train_rnn(["tone_16bit_spectrum.csv", "tone_24bit_spectrum.csv"],
                                hidden_size=32, num_layers=1, num_classes=3, epochs=10)
"""

import numpy as np
from pathlib import Path

# scikit-learn MLP
from sklearn.neural_network import MLPClassifier
import joblib

# PyTorch for NN and RNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----------------------------
# Fixed suffix -> index mapping
# (keeps labels small and consistent for PyTorch losses)
# ----------------------------
SUFFIX_TO_INDEX = {
    "16bit": 0,
    "24bit": 1,
    "float32": 2,
}

# ----------------------------
# MLP (existing, scikit-learn)
# ----------------------------
def load_spectra(files):
    """
    Load spectra CSVs as numpy arrays for scikit-learn MLP.
    Assumes the first row is a header and data starts from row 1.
    Returns:
        X (ndarray): shape (batch, features)
        y (ndarray): shape (batch,)
    """
    spectra = []
    labels = []
    for f in files:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        # If it's 2D, flatten to 1D feature vector
        data_flat = data.flatten()
        spectra.append(data_flat)
        # Extract suffix, e.g., tone_16bit_spectrum.csv -> '16bit'
        try:
            suffix = Path(f).stem.split('_')[1]
        except Exception:
            suffix = str(Path(f).stem)
        # Map to small integer index if possible; fall back to 0
        label = SUFFIX_TO_INDEX.get(suffix, 0)
        labels.append(label)
    X = np.vstack(spectra).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y

def train_MLP(spectrum_files):
    """
    Train a scikit-learn MLP on flattened spectral data.
    This preserves the original MLP workflow but uses stable label mapping.
    Saves model to 'ai_model.pkl'.
    Returns:
        model (MLPClassifier)
    """
    X, y = load_spectra(spectrum_files)
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
    model.fit(X, y)
    joblib.dump(model, "ai_model.pkl")
    return model

# ----------------------------
# PyTorch feedforward NN (new)
# ----------------------------
def load_spectra_nn(files):
    """
    Load spectra CSVs as tensors for PyTorch feedforward NN.
    Returns:
        X (torch.FloatTensor): shape (batch, features)
        y (torch.LongTensor): shape (batch,)
        input_size (int): number of features
    """
    spectra = []
    labels = []
    for f in files:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        data_flat = data.flatten()
        spectra.append(data_flat.astype(np.float32))
        try:
            suffix = Path(f).stem.split('_')[1]
        except Exception:
            suffix = str(Path(f).stem)
        labels.append(SUFFIX_TO_INDEX.get(suffix, 0))
    X = np.vstack(spectra).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    input_size = X.shape[1]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size

class SpectrumNN(nn.Module):
    """
    Simple PyTorch feedforward NN for flattened spectral input.
    """
    def __init__(self, input_size, hidden_size=128, num_classes=3):
        super(SpectrumNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_nn(spectrum_files, hidden_size=128, lr=1e-3, epochs=50):
    """
    Train a PyTorch feedforward SpectrumNN on flattened spectral data.
    Auto-detects input_size and number of classes from files (using SUFFIX_TO_INDEX).
    Saves model state to 'nn_model.pt'.
    Returns:
        model (SpectrumNN)
    """
    X, y, input_size = load_spectra_nn(spectrum_files)
    # Ensure classes cover mapping (safe)
    num_classes = max(1, len(set(SUFFIX_TO_INDEX.values())))
    model = SpectrumNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"NN Epoch [{epoch+1}/{epochs}], Loss={loss.item():.6f}")

    torch.save(model.state_dict(), "nn_model.pt")
    return model

# ----------------------------
# RNN (existing, PyTorch)
# ----------------------------
def load_spectra_rnn(files):
    """
    Load spectra CSVs as sequences for RNN input.
    Automatically infers input_size from CSV shape.
    Returns:
        X (torch.FloatTensor): shape (batch, seq_len, features)
        y (torch.LongTensor): shape (batch,)
        input_size (int): number of features per time step
    """
    spectra = []
    labels = []
    for f in files:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        # Ensure 2D: (seq_len, features)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        spectra.append(data.astype(np.float32))
        try:
            suffix = Path(f).stem.split('_')[1]
        except Exception:
            suffix = str(Path(f).stem)
        labels.append(SUFFIX_TO_INDEX.get(suffix, 0))
    X = np.stack(spectra)  # (batch, seq_len, features)
    y = np.array(labels, dtype=np.int64)
    input_size = X.shape[2]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size

class SpectrumRNN(nn.Module):
    """
    LSTM-based RNN for spectral sequences.
    Kept compatible with existing tests / usage.
    """
    def __init__(self, input_size, hidden_size=50, num_layers=1, num_classes=3):
        super(SpectrumRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq_len, hidden_size)
        out = out[:, -1, :]          # take last time step
        out = self.fc(out)
        return out

def train_rnn(spectrum_files, hidden_size=50, num_layers=1, num_classes=3,
              lr=0.001, epochs=50):
    """
    Train an LSTM RNN on spectral sequences.
    Auto-detects input_size from CSVs.
    Saves model state to 'rnn_model.pt'.
    Returns:
        model (SpectrumRNN)
    """
    X, y, input_size = load_spectra_rnn(spectrum_files)
    model = SpectrumRNN(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"RNN Epoch [{epoch+1}/{epochs}], Loss={loss.item():.6f}")

    torch.save(model.state_dict(), "rnn_model.pt")
    return model

