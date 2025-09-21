# Audio-Image-Video-Toolkit-AI / Image
**Author:** Julia Wen (wendigilane@gmail.com)

## Overview
This folder contains scripts for crescent moon detection and synthetic image generation.

### Detection Approaches
- `predict_crescent.py` – Python SVM with HOG + augmentation  
- `predict_crescent_pytorch.py` – PyTorch-based SVM classifier using HOG features  
- `predict_crescent_pytorch_cnn.py` – PyTorch CNN classifier with augmentation, CPU/GPU switching, and optional temperature scaling for new images  
- `predict_crescent_tf.py` – TensorFlow CNN classifier built with tf.keras; saves trained model as a `.keras` file; supports YAML config, saves labeled `_tf.png` images and results text in `test_output/`  
- `predict_crescent_tf_classic.py` – TensorFlow wrapper around classic HOG + SVM approach; saves model as a `.h5` file  
- `predict_crescent_vit.py` – Vision Transformer-based crescent prediction  
- `detect_crescent_classical.py` – Classical HOG + SVM approach  
- `app_predict_crescent_tf.py` – **Streamlit app for TensorFlow CNN**, supports live training charts, external image upload, labeled image output (`_tf.png`), and results saved in `test_output/`  

### Synthetic Image Generation
- `generate_crescent_images.py` – Stable Diffusion-based crescent/no-crescent images (CPU/CUDA/MPS), with internal negative prompts and logging. Includes Safety Checker note for quick local testing.

---

## Quick Start / Installation

1. **Create virtual environment**  
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```  
Dependencies include: `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `joblib`, `torch`, `torchvision`, `torchaudio`, `tensorflow`, `opencv-python`, `imageio`, `imageio-ffmpeg`, `soundfile` (optional), `timm` (optional), `streamlit`.

---

## Quick Start Usage Examples

### Dataset Structure
```
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
Images can be `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tif`, or `.tiff`. Place images for each class in the corresponding folder.

### Train and Predict

```bash
# PyTorch + SVM
python3 predict_crescent_pytorch.py

# PyTorch CNN
python3 predict_crescent_pytorch_cnn.py

# TensorFlow CNN
python3 predict_crescent_tf.py
python3 predict_crescent_tf_classic.py

# Streamlit App (TensorFlow CNN)
streamlit run app_predict_crescent_tf.py
```

### Predict a Single New Image
```bash
python3 predict_crescent_pytorch.py --image path/to/image.jpg
python3 predict_crescent_pytorch_cnn.py --image path/to/image.jpg
python3 predict_crescent_tf.py --image path/to/image.keras
python3 predict_crescent_tf_classic.py --image path/to/image.h5
```

### Disable Plotting
```bash
python3 predict_crescent_pytorch.py --no-display
python3 predict_crescent_pytorch_cnn.py --no-display
python3 predict_crescent_tf.py --no-display
python3 predict_crescent_tf_classic.py --no-display
```

### Vision Transformer Demo
```bash
python3 predict_crescent_vit.py
python3 predict_crescent_vit.py --image path/to/image.jpg
```

### Generative AI Image Generation
```bash
python3 generate_crescent_images.py
```

---

## How It Works

- Images are read and resized to 128×128 pixels.  
- Color or grayscale conversion depending on the script.  
- HOG feature extraction for SVM-based approaches.  
- Data augmentation applied where enabled.  
- Feature scaling using StandardScaler.  
- Linear SVM, PyTorch logistic regression, or CNN trained end-to-end.  
- Predictions output with probability/confidence scores.  
- Original images displayed with predicted labels and probabilities; augmented images used for training only.  
- **TensorFlow CNN (`predict_crescent_tf.py`)** now supports:
  - YAML configuration (`config_predict_crescent_tf.yaml`)  
  - Output results saved in `test_output/predict_crescent_tf_result.txt`  
  - Labeled images with `_tf.png` suffix for reference  
- **Streamlit app (`app_predict_crescent_tf.py`)**:
  - Live training plots via `st.line_chart`  
  - Upload and predict new images  
  - Saves `_tf.png` labeled images and results text automatically  

---

## Model and Scaler Persistence

### PyTorch SVM
```python
import torch
model = torch.load("crescent_moon_logreg.pt")
```
### Scaler
```python
import joblib
scaler = joblib.load("crescent_moon_scaler.pkl")
```
### PyTorch CNN
```python
# Saved as crescent_cnn_best.pth
import torch
model = torch.load("crescent_cnn_best.pth")
```
### TensorFlow CNN (tf.keras)
```python
from tensorflow.keras.models import load_model
cnn_model = load_model("crescent_moon_cnn.keras")
```
### TensorFlow classic HOG + SVM
```python
from tensorflow.keras.models import load_model
classic_model = load_model("crescent_moon_tf_classic.h5")
```
### Scikit-learn SVM
```python
import joblib
svm = joblib.load("crescent_moon_model.pkl")
scaler = joblib.load("crescent_moon_scaler.pkl")
```

---

## Notes

- Some source code in this repository was assisted with AI tools.  
- AI assistance was used as a development aid only; core algorithmic logic and dataset-specific handling were implemented and verified manually.  
- Streamlit app allows interactive training, predictions, and saves outputs automatically in `test_output/`.  

---

## License

MIT License

