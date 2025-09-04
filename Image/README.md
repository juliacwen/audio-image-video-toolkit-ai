# Audio-Image-Video-Toolkit-AI / Image

**Author:** Julia Wen (<wendigilane@gmail.com>)

## Overview

This folder contains scripts for **crescent moon detection and synthetic image generation**:

- **Detection approaches:**
  - `predict_crescent.py` – Python SVM with HOG + augmentation  
  - `predict_crescent_pytorch.py` – PyTorch-based SVM classifier using HOG features
  - `predict_crescent_tf.py` – TensorFlow CNN classifier with data augmentation
  - `detect_crescent_classical.py` – Classical HOG + SVM approach
  - `predict_crescent_vit.py` – Vision Transformer-based crescent prediction

- **Synthetic image generation:**  
  - `generate_crescent_images.py` – Stable Diffusion-based crescent/no-crescent images (CPU/CUDA/MPS), with internal negative prompts and logging. Includes **Safety Checker note for quick local testing**.

---

## Quick Start / Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

- Dependencies include: `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `joblib`, `torch`, `torchvision`, `torchaudio`, `tensorflow`, `opencv-python`, `imageio`, `imageio-ffmpeg`, `soundfile` (optional), `timm` (optional).

---

## Quick Start Usage Examples

```bash
# Prepare dataset
#   dataset/crescent/  → crescent moon images
#   dataset/no_crescent/ → images without crescent

# Train and predict on dataset using PyTorch + SVM
python3 predict_crescent_pytorch.py

# Train and predict on dataset using TensorFlow + CNN
python3 predict_crescent_tf.py

# Predict a single new image
python3 predict_crescent_pytorch.py --image path/to/image.jpg
python3 predict_crescent_tf.py --image path/to/image.jpg

# Disable plotting
python3 predict_crescent_pytorch.py --no-display
python3 predict_crescent_tf.py --no-display

# Vision Transformer Demo
python3 predict_crescent_vit.py
python3 predict_crescent_vit.py --image path/to/image.jpg

# Generative AI Image Generation
python3 generate_crescent_images.py
```

---

## Crescent Moon Detection

This repository provides **six approaches** for detecting crescent moons in images:

1. `predict_crescent.py` – Augmented HOG + SVM (scikit-learn).
2. `predict_crescent_pytorch.py` – PyTorch-based SVM classifier using augmented HOG features.
3. `predict_crescent_tf.py` – TensorFlow CNN classifier with data augmentation.
4. `detect_crescent_classical.py` – Classical HOG + SVM approach.
5. `predict_crescent_vit.py` – Vision Transformer-based crescent prediction.
6. `generate_crescent_images.py` – Synthetic crescent/no-crescent image generator using Stable Diffusion.

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
- Extracts **HOG features** from images, with optional augmentation.
- Scales features with **StandardScaler**.
- Trains a **logistic regression / SVM classifier** (PyTorch implementation).
- Only original images are displayed.
- Supports **external image prediction** with `--image`.
- Saves **model** and **scaler** (`crescent_moon_logreg.pt` and `crescent_moon_scaler.pkl`).
- Optional plotting can be disabled with `--no-display`.

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

## 5. `predict_crescent_vit.py` (Vision Transformer)

- Uses a **Vision Transformer (ViT)** to predict crescent moons in images.
- Supports **predicting a single external image**.
- Optional display or logging handled internally.

---

## 6. `generate_crescent_images.py` (Synthetic Image Generation)

- Generates synthetic **crescent and no-crescent moon images** using Stable Diffusion.
- Supports CPU, CUDA, and Apple MPS devices.
- Uses **internal negative prompts** to reduce full moon outputs.
- Logs image metadata (seed, filename, label) in `generation_log.csv`.
- Handles Ctrl-C gracefully.

**Safety Checker Disabled:**

- The script does **not enable the Hugging Face safety checker**.
- You may see a warning:

```
You have disabled the safety checker...
```

- This is **for quick local testing**.
- If you intend to **share images publicly or use them in a service**, you **must enable a safety checker** or filter images manually, according to the Stable Diffusion license.
- Reference: [Hugging Face Diffusers Safety Checker](https://github.com/huggingface/diffusers/pull/254)

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

