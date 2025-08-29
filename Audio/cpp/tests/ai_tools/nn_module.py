# ai_tools/nn_module.py
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib

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
        label = int(f.stem.split('_')[1].replace('bit',''))
        labels.append(label)
    X = np.vstack(spectra)
    y = np.array(labels)
    return X, y

def train_nn(spectrum_files):
    X, y = load_spectra(spectrum_files)
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
    model.fit(X, y)
    # Save model
    joblib.dump(model, "ai_model.pkl")
    return model

