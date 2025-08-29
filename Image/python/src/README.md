# Audio-Image-Video-Toolkit-AI/image

**Author:** Julia Wen (<wendigilane@gmail.com>)

# Crescent Moon Detection

This project provides two approaches for detecting crescent moons in images (both scripts live in the same repository):

1. `predict_crescent.py` — a modern approach using data augmentation, feature extraction, and a linear SVM classifier.  
   - **Supports** predicting a single external image via the `--image` parameter.
   - Uses optional plotting (`matplotlib`) and can save/load the trained model and scaler.

2. `detect_crescent_classical.py` — a simpler "classical" approach using grayscale conversion, HOG (Histogram of Oriented Gradients) features, and a linear SVM.  
   - **Does NOT** accept an external `--image` parameter; it trains and predicts only on the images loaded from the dataset directory.
   - Prints predictions for the dataset it loads; no model persistence or external-image prediction by default.

Both scripts train on a dataset of `crescent` vs `no_crescent` images and output confidence scores for their predictions.

## Algorithmic Differences

- **predict_crescent.py**  
  - Uses **PyTorch** and **torchvision** for image preprocessing and augmentation.  
  - Extracts features (e.g., HOG-like embeddings or torchvision transforms).  
  - Trains a **linear SVM** classifier from `scikit-learn`.  
  - Includes **augmentation**, **optional plotting**, and **external image prediction**.  
  - Can save/load models and scalers (via `joblib` if enabled).

- **detect_crescent_classical.py**  
  - Uses **scikit-image** for grayscale conversion, resizing, and **HOG feature extraction**.  
  - Trains a **linear SVM** from `scikit-learn`.  
  - No augmentation, no external-image prediction, no model saving.  
  - Outputs predictions directly on the dataset images.

## Requirements

Both scripts share most dependencies, but differ slightly:

- **Shared**
  - `numpy`
  - `scikit-learn`
  - `scikit-image`
  - `matplotlib` (for optional plotting)
  - `opencv-python` (image handling)
  - `pandas`, `scipy` (general utilities)

- **predict_crescent.py only**
  - `torch`, `torchvision`, `torchaudio` (augmentation & transforms)
  - `tensorflow` (optional, depending on environment)
  - `timm` (for potential feature extraction backbones)
  - `imageio`, `imageio-ffmpeg`, `soundfile` (extra media support)

- **detect_crescent_classical.py only**
  - No extra packages beyond **scikit-image** + **scikit-learn**

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

Notes:
- Images can be `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tif`, or `.tiff`.
- Place images for each class in the corresponding folder.

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

Notes:
- `scikit-learn` includes its own `joblib`, so no separate install is needed.
- `matplotlib` is only required if you want to visualize images during training.

## Usage

```text
# Train and predict on the dataset using the main script
python3 predict_crescent.py

# Predict a single external image
python3 predict_crescent.py --image path/to/image.jpg

# Disable image display during training or prediction
python3 predict_crescent.py --no-display
python3 predict_crescent.py --no-display --image path/to/image.jpg

# Run the classical HOG-based script (predicts only on dataset images)
python3 detect_crescent_classical.py

Notes:
- predict_crescent.py supports augmentation, plotting, and saving/loading model and scaler.
- detect_crescent_classical.py is simpler: no augmentation, no external-image prediction, prints predictions only.

## Model Saving and Loading

```text
# predict_crescent.py
- The trained SVM model is saved as `crescent_moon_model.pkl`.
- The feature scaler is saved as `crescent_moon_scaler.pkl`.
- These can be loaded later for prediction on new images:

import joblib
svm = joblib.load("crescent_moon_model.pkl")
scaler = joblib.load("crescent_moon_scaler.pkl")

# detect_crescent_classical.py
- No model or scaler is saved to disk.
- Predictions are made directly on the loaded dataset images only.

## License
MIT License

## Project Overview
This project includes **two approaches** to crescent detection:
### 1. `predict_crescent.py` (Augmented Deep Features + SVM)
- Uses **PyTorch**, **TensorFlow**, and **OpenCV** for image preprocessing and augmentation.
- Extracts **deep features** from images and trains a **linear SVM classifier**.
- Supports **data augmentation** to increase robustness.
- Can take an external image via `--image` parameter.
- Optional plotting during training (disable with `--no-display`).

### 2. `detect_crescent_classical.py` (Classical HOG + SVM)
- Uses **scikit-image** to extract **Histogram of Oriented Gradients (HOG)** features.
- Trains a **linear SVM classifier** directly on these hand-crafted features.
- No deep learning frameworks involved (lighter and faster).
- Simpler pipeline, but usually less accurate on complex datasets.

### Why "Classical"?
The **classical script** is named so because it relies on **traditional computer vision techniques (HOG + SVM)** instead of **deep learning or neural network features**. It’s easier to run with fewer dependencies, but generally less powerful than the feature-augmented approach.

## Requirements

torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
opencv-python==4.10.0.84
matplotlib
numpy
pandas
scipy
timm
imageio
imageio-ffmpeg
soundfile
tensorflow==2.20.0
scikit-learn
scikit-image

### Installing dependencies

python -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate for Windows
pip install -r requirements.txt

### Notes

- scikit-learn includes the internal version of joblib, so it doesn’t need to be installed separately.
- matplotlib is used for optional image plotting.

## Dataset Structure

dataset/
├── crescent/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── no_crescent/
    ├── image1.jpg
    ├── image2.png
    └── ...
Images can be .jpg, .png, .jpeg, .bmp, .tif, or .tiff.
Each class folder contains images corresponding to that class.

## Crescent Detection Scripts

1. predict_crescent.py:
   - Loads images from the dataset and resizes to 128x128 pixels.
   - Converts images to grayscale.
   - Extracts HOG features.
   - Scales features using StandardScaler.
   - Optional image augmentation (flips, rotations, noise).
   - Trains a linear SVM classifier with probability estimates.
   - Can predict a single new image using --image parameter.
   - Optionally displays images and predictions with matplotlib.
   - Saves trained model and scaler to disk using joblib.

2. detect_crescent_classical.py:
   - Loads images from the dataset and resizes to 128x128 pixels.
   - Converts images to grayscale.
   - Extracts HOG features.
   - Scales features using StandardScaler.
   - Trains a linear SVM classifier with probability estimates.
   - Predicts on the same dataset and prints predictions with confidence scores.
   - No image augmentation.
   - Does not save model or scaler.
   - No optional input for single new image.
   - No plotting; prints predictions only.

## Models and Packages Used

1. predict_crescent.py:
   - Model: Linear SVM with probability estimates
   - Feature extraction: HOG
   - Feature scaling: StandardScaler
   - Packages: numpy, scikit-image, scikit-learn, matplotlib (optional), joblib

2. detect_crescent_classical.py:
   - Model: Linear SVM with probability estimates
   - Feature extraction: HOG
   - Feature scaling: StandardScaler
   - Packages: numpy, scikit-image, scikit-learn

## Usage

### Training and Predicting on Original Dataset

python predict_crescent.py
- Loads images from the dataset with augmentation.
- Trains the linear SVM classifier.
- Displays predictions for the original images with probabilities.

### Predicting a New Image

python predict_crescent.py --image path_to_image.jpg
- Predicts whether the new image contains a crescent moon.
- Displays the image and its predicted probability.

### Disable Display (No Plots)

python predict_crescent.py --no-display
python predict_crescent.py --no-display --image path_to_image.jpg
- Runs predictions without plotting any images.

## Model and Scaler Saving

- The trained SVM model is saved as crescent_moon_model.pkl
- The feature scaler is saved as crescent_moon_scaler.pkl
- These files can be loaded later for prediction on new images without retraining

import joblib
svm = joblib.load("crescent_moon_model.pkl")
scaler = joblib.load("crescent_moon_scaler.pkl")

## How It Works

1. Images are read and converted to grayscale.
2. Resized to 128x128 pixels.
3. HOG (Histogram of Oriented Gradients) features are extracted for each image.
4. Feature vectors are scaled using StandardScaler.
5. A linear SVM is trained on the scaled features.
6. Prediction probabilities are calculated using svm.predict_proba().
7. Optional augmentation increases dataset size (flips, rotations, noise).


## License
MIT License
