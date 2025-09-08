# Audio-Image-Video-Toolkit-AI / Image

**Author:** Julia Wen (<wendigilane@gmail.com>)

## Overview

This folder contains scripts for **crescent moon detection and synthetic image generation**:

- **Detection approaches:**
  - `predict_crescent.py` – Python SVM with HOG + augmentation  
  - `predict_crescent_pytorch.py` – PyTorch-based SVM classifier using HOG features
  - `predict_crescent_pytorch_cnn.py` – PyTorch CNN classifier with augmentation, CPU/GPU switching, and optional temperature scaling for new images
  - `predict_crescent_tf.py` – TensorFlow CNN classifier built with tf.keras; saves trained model as a `.keras` file
  - `predict_crescent_tf_classic.py` – TensorFlow wrapper around classic HOG + SVM approach; saves model as a `.h5` file
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

# Train and predict on dataset using PyTorch CNN
python3 predict_crescent_pytorch_cnn.py

# Train and predict on dataset using TensorFlow CNN
python3 predict_crescent_tf.py
python3 predict_crescent_tf_classic.py

# Predict a single new image
python3 predict_crescent_pytorch.py --image path/to/image.jpg
python3 predict_crescent_pytorch_cnn.py --image path/to/image.jpg
python3 predict_crescent_tf.py --image path/to/image.keras
python3 predict_crescent_tf_classic.py --image path/to/image.h5

# Disable plotting
python3 predict_crescent_pytorch.py --no-display
python3 predict_crescent_pytorch_cnn.py --no-display
python3 predict_crescent_tf.py --no-display
python3 predict_crescent_tf_classic.py --no-display

# Vision Transformer Demo
python3 predict_crescent_vit.py
python3 predict_crescent_vit.py --image path/to/image.jpg

# Generative AI Image Generation
python3 generate_crescent_images.py
```

---

## Crescent Moon Detection

This repository provides **seven approaches** for detecting crescent moons in images:

1. `predict_crescent.py` – Augmented HOG + SVM (scikit-learn).
2. `predict_crescent_pytorch.py` – PyTorch-based SVM classifier using augmented HOG features.
3. `predict_crescent_pytorch_cnn.py` – PyTorch CNN with data augmentation, CPU/GPU switching, and new-image temperature scaling.
4. `predict_crescent_tf.py` – TensorFlow CNN classifier built with tf.keras; saves trained model as a `.keras` file.
5. `predict_crescent_tf_classic.py` – TensorFlow wrapper around classic HOG + SVM approach; saves model as a `.h5` file.
6. `detect_crescent_classical.py` – Classical HOG + SVM approach.
7. `predict_crescent_vit.py` – Vision Transformer-based crescent prediction.
8. `generate_crescent_images.py` – Synthetic crescent/no-crescent image generator using Stable Diffusion.

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

**PyTorch CNN:**
```python
# Saved as crescent_cnn_best.pth
import torch
model = torch.load("crescent_cnn_best.pth")
```

**TensorFlow CNN (`tf.keras`):**
```python
from tensorflow.keras.models import load_model
cnn_model = load_model("crescent_moon_cnn.keras")
```

**TensorFlow classic HOG + SVM:**
```python
from tensorflow.keras.models import load_model
classic_model = load_model("crescent_moon_tf_classic.h5")
```

**Scikit-learn SVM:**
```python
import joblib
svm = joblib.load("crescent_moon_model.pkl")
scaler = joblib.load("crescent_moon_scaler.pkl")
```

---

## Notes

- Some source code in this repository was **gassisted with AI tools**.
- AI assistance was used as a **development aid** only; core algorithmic logic and dataset-specific handling were implemented and verified manually.

---

## License

MIT License

