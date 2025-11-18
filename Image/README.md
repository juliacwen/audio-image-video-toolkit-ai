# Audio-Image-Video-Toolkit-AI / Image
**Author:** Julia Wen (wendigilane@gmail.com)

## Overview
This folder contains scripts for crescent moon detection and synthetic image generation.

- **Detection Approaches:** Python scripts (SVM, PyTorch CNN, TensorFlow CNN, HOG, ViT) for crescent detection.
- **TypeScript & JavaScript Demos:** Frontend examples demonstrating crescent detection using `Image/typescript/web` or `Image/javascript/web`.

---

## Directory Layout
```text
Image/
  python/
    src/
      predict_crescent.py
      predict_crescent_pytorch.py
      predict_crescent_pytorch_cnn.py
      predict_crescent_tf.py
      predict_crescent_tf_classic.py
      predict_crescent_vit.py
      detect_crescent_classical.py
      app_predict_crescent_tf.py
      generate_crescent_images.py
      server.py
  typescript/
    crescentDemo.ts
    web/
      App.tsx
      main.tsx
      index.html
      package.json
      vite.config.ts
  javascript/
    web/
      App.jsx
      main.jsx
      index.html
      package.json
      vite.config.js
  dataset/
    crescent/
    no_crescent/
  test_output/
```

---

## Quick Start / Installation

### 1. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```
Dependencies include: `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `joblib`, `torch`, `torchvision`, `torchaudio`, `tensorflow`, `opencv-python`, `imageio`, `imageio-ffmpeg`, `soundfile` (optional), `timm` (optional), `streamlit`.

---

## Dataset Structure
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

---

## Python Scripts (Image/python/src)

### Detection Approaches
- `predict_crescent.py` – Python SVM with HOG + augmentation  
- `predict_crescent_pytorch.py` – PyTorch SVM classifier using HOG features  
- `predict_crescent_pytorch_cnn.py` – PyTorch CNN classifier with augmentation, CPU/GPU switching, optional temperature scaling  
- `predict_crescent_tf.py` – TensorFlow CNN classifier built with tf.keras; YAML config supported, outputs results in `test_output/`  
- `predict_crescent_tf_classic.py` – TensorFlow wrapper around HOG + SVM approach  
- `predict_crescent_vit.py` – Vision Transformer-based crescent prediction  
- `detect_crescent_classical.py` – Classical HOG + SVM approach  
- `app_predict_crescent_tf.py` – **Streamlit app for TensorFlow CNN**; live training charts, external image upload, labeled image output (`_tf.png`), results saved in `test_output/`  
- `server.py` – FastAPI server for live crescent detection, using SVM + HOG; handles file upload and returns prediction with confidence
- `generate_crescent_images.py` – Stable Diffusion-based crescent/no-crescent image generation (CPU/CUDA/MPS), with internal negative prompts and logging

---

## TypeScript Demo (Image/typescript/web)
- `crescentDemo.ts` – Demonstrates crescent detection logic in TypeScript; simulates predictions, logs results in console. Uses dataset under `Image/typescript/web/public/dataset/`.

### Run TypeScript Demo
```bash
cd Image/typescript/web
npm install
npm run dev
```
- Open the URL Vite prints (e.g., `http://localhost:5173/`) in your browser.  
- Click **Run Detection** to simulate crescent detection results.

---

## JavaScript Demo (Image/javascript/web)
- `App.jsx` – React + JavaScript frontend for crescent detection; communicates with FastAPI backend at `http://127.0.0.1:8000/detect`.  
- `main.jsx`, `index.html`, `package.json`, `vite.config.js` – Main entry and configuration files for JavaScript frontend.  
- Uses dataset under `Image/javascript/web/public/dataset/`.

### Run JavaScript Demo
```bash
cd Image/javascript/web
npm install
npm run dev
```
- Open the URL Vite prints (e.g., `http://localhost:5173/`) in your browser.  
- Click **Run Detection** to see real predictions from your backend.  
- The page title can indicate whether the TypeScript or JavaScript frontend is running.

---

## Quick Start Usage Examples (Python)

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

# FastAPI Server
python3 server.py
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
- Python scripts reside under `Image/python/src`.  
- TypeScript demo (`crescentDemo.ts`) is under `Image/typescript` and uses `Image/typescript/web` for datasets.  
- JavaScript frontend uses `Image/javascript/web`.  
- Streamlit app allows interactive training and saves outputs automatically in `test_output/`.  
- FastAPI server supports live prediction via file upload.  
- Add `console.log` or modify page title in the frontend to distinguish TypeScript vs JavaScript demos.

---

## License
MIT License

