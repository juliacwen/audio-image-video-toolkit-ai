#!/usr/bin/env python3
# ai_tools/nn_module.py
"""
ai_tools/nn_module.py - Spectra NN module with optional RNN (auto-detect sequence length)

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-11

Example usage:

# ----------------------------
# Train MLP (existing workflow)
# ----------------------------
from nn_module import train_nn

spectrum_files = ["tone_16bit.csv", "tone_32bit.csv"]  # replace with your files
mlp_model = train_nn(spectrum_files)
print("MLP model trained and saved as ai_model.pkl")

# ----------------------------
# Train RNN (new workflow)
# ----------------------------
from nn_module import train_rnn

rnn_model = train_rnn(
    spectrum_files,
    hidden_size=64,
    num_layers=2,
    num_classes=16,  # adjust according to your labels
    epochs=100
)
print("RNN model trained and saved as rnn_model.pt")
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
from pathlib import Path

# For RNN
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# MLP (existing)
# ----------------------------
def load_spectra(files):
    """
    Load spectra CSVs as numpy arrays.
    Assumes the first row is a header and data starts from row 1.
    """
    spectra = []
    labels = []
    for f in files:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        spectra.append(data.flatten())
        # Use filename to generate labels (example: tone_16bit.csv → 16)
        label = int(Path(f).stem.split('_')[1].replace('bit',''))
        labels.append(label)
    X = np.vstack(spectra)
    y = np.array(labels)
    return X, y

def train_nn(spectrum_files):
    """
    Train a feedforward MLP on flattened spectral data.
    """
    X, y = load_spectra(spectrum_files)
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
    model.fit(X, y)
    # Save model
    joblib.dump(model, "ai_model.pkl")
    return model

# ----------------------------
# RNN
# ----------------------------
def load_spectra_rnn(files):
    """
    Load spectra CSVs as sequences for RNN input.
    Automatically infers input_size from CSV shape.
    Returns: X (batch, seq_len, features), y (labels), input_size
    """
    spectra = []
    labels = []
    for f in files:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        spectra.append(data)  # keep 2D (seq_len, features)
        # Generate label from filename, e.g., tone_16bit.csv -> 16
        label = int(Path(f).stem.split('_')[1].replace('bit',''))
        labels.append(label)
    X = np.stack(spectra)  # shape: (batch, seq_len, features)
    y = np.array(labels)
    input_size = X.shape[2]  # features per time step
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), input_size

class SpectrumRNN(nn.Module):
    """
    LSTM-based RNN for spectral sequences.
    """
    def __init__(self, input_size, hidden_size=50, num_layers=1, num_classes=10):
        super(SpectrumRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq_len, hidden_size)
        out = out[:, -1, :]          # take last time step
        out = self.fc(out)
        return out

def train_rnn(spectrum_files, hidden_size=50, num_layers=1, num_classes=10,
              lr=0.001, epochs=50):
    """
    Train an LSTM RNN on spectral sequences.
    Auto-detects input_size from CSVs.
    """
    X, y, input_size = load_spectra_rnn(spectrum_files)
    model = SpectrumRNN(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "rnn_model.pt")
    return model

