# Audio-Image-Video-Toolkit-AI / Image

**Author:** Julia Wen (<wendigilane@gmail.com>)

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset
#   dataset/crescent/  → crescent moon images
#   dataset/no_crescent/ → images without crescent

# 4. Train and predict on dataset using PyTorch + SVM
python3 predict_crescent_pytorch.py

# 5. Train and predict on dataset using TensorFlow + CNN
python3 predict_crescent_tf.py

# 6. Predict a single new image with PyTorch
python3 predict_crescent_pytorch.py --image path/to/image.jpg

# 7. Predict a single new image with TensorFlow
python3 predict_crescent_tf.py --image path/to/image.jpg

# 8. Disable plotting
python3 predict_crescent_pytorch.py --no-display
python3 predict_crescent_tf.py --no-display
```

**Notes:**

- Only **original images** are displayed; augmented images are used for training.
- Trained **PyTorch SVM model** and **scaler** are saved as:
  - `crescent_moon_logreg.pt`
  - `crescent_moon_scaler.pkl`
- Trained **TensorFlow CNN model** is saved as:
  - `crescent_moon_cnn.h5`
- These models can be used for new predictions without retraining.

---

## Crescent Moon Detection

This repository provides **three approaches** for detecting crescent moons in images:

1. **`predict_crescent.py`** – Augmented HOG + SVM (scikit-learn).
2. **`predict_crescent_pytorch.py`** – PyTorch-based SVM classifier using augmented HOG features.
3. **`predict_crescent_tf.py`** – TensorFlow CNN classifier with data augmentation.
4. **`detect_crescent_classical.py`** – Classical HOG + SVM approach.

All scripts train on a dataset of `crescent` vs `no_crescent` images and output confidence scores.

---

## 1. `predict_crescent.py` (Python SVM)

- Uses **scikit-image** for grayscale conversion, resizing, and **HOG feature extraction**.
- Applies **data augmentation** (flips, rotations, noise) for training.
- Scales features with **StandardScaler**.
- Trains a **linear SVM classifier** with probability estimates.
- Only **original images** are displayed with **color**.
- Supports **predicting a single external image** with `--image`.
- Optional plotting (`matplotlib`) can be disabled with `--no-display`.
- Saves **model** and **scaler** to disk (`crescent_moon_model.pkl` and `crescent_moon_scaler.pkl`).

---

## 2. `predict_crescent_pytorch.py` (PyTorch SVM)

- Uses **PyTorch** for handling data and training.
- Extracts **HOG features** with augmentation for training.
- Scales features with **StandardScaler**.
- Trains a **logistic regression/SVM** classifier in PyTorch.
- Only original images are displayed.
- Supports **external image prediction** with `--image`.
- Saves **model** and **scaler** (`crescent_moon_logreg.pt` and `crescent_moon_scaler.pkl`).

---

## 3. `predict_crescent_tf.py` (TensorFlow CNN)

- Uses **TensorFlow/Keras CNN** for end-to-end learning.
- Applies **data augmentation** (rotations, flips, noise) to the dataset.
- Trains on color images directly.
- Only **original images** are displayed.
- Supports **external image prediction** with `--image`.
- Saves trained model to `crescent_moon_cnn.h5`.
- Optional plotting can be disabled with `--no-display`.

---

## 4. `detect_crescent_classical.py` (Classical HOG + SVM)

- Uses **scikit-image** to extract HOG features.
- Scales features using **StandardScaler**.
- Trains a **linear SVM classifier**.
- **No augmentation** applied.
- Predicts only on the dataset images; no external image prediction.
- Prints predictions with confidence scores; no plotting.
- Does **not save model or scaler**.

---

## Dataset Structure

```text
dataset/
├── crescent/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── no_crescent/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

- Images can be `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tif`, or `.tiff`.
- Place images for each class in the corresponding folder.

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**Dependencies include:**

- `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `joblib`
- `torch`, `torchvision`, `torchaudio` (PyTorch)
- `tensorflow` (TensorFlow)
- `opencv-python`, `imageio`, `imageio-ffmpeg`, `soundfile` (optional)
- `timm` (optional for deep feature extraction)

---

## How It Works

1. Images read and resized to 128×128 pixels.
2. Color or grayscale conversion depending on script.
3. HOG feature extraction for SVM-based approaches.
4. Data augmentation applied where enabled.
5. Feature scaling using StandardScaler.
6. Linear SVM or PyTorch logistic regression trained on features, or CNN trained end-to-end.
7. Predictions output with probability/confidence scores.
8. Original images displayed with predicted labels and probabilities; augmented images used for training only.

---

## Model and Scaler Persistence

**PyTorch SVM:**
```python
import torch
model = torch.load("crescent_moon_logreg.pt")
```
**Scaler:**
```python
import joblib
scaler = joblib.load("crescent_moon_scaler.pkl")
```

**TensorFlow CNN:**
```python
from tensorflow.keras.models import load_model
cnn_model = load_model("crescent_moon_cnn.h5")
```

**Scikit-learn SVM:**
```python
import joblib
svm = joblib.load("crescent_moon_model.pkl")
scaler = joblib.load("crescent_moon_scaler.pkl")
```

---

## License

MIT License

