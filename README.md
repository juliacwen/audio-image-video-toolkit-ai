# Audio-Image-Video-Toolkit-AI
**Author:** Julia Wen (<wendigilane@gmail.com>)

## Project Overview
This repository contains projects for **Audio**, **Image**, and **Video** processing, organized into separate top-level directories. Each area contains source code (C++ and Python) and tests (Pytest and GoogleTest) intended for development and experimentation.

## Features

**Audio**: Converting WAV to CSV, generating FFT spectra with multiple window types, running end-to-end AI-assisted FFT workflows that generate test WAVs, train a small MLP on synthetic spectra, and predict tone probabilities, performing automated FFT windowing tests with rectangular, Hann, Hamming, and Blackman windows.

**Image**: Crescent detection in images, with multiple classical and machine learning approaches and dataset generation. Supports PyTorch CNN, tf.keras CNN, HOG + SVM (classic), and Vision Transformer approaches.

**Video**: C++ modules for motion estimation, video encoding, depth estimation, trajectory analysis, plus Python scripts for video depth estimation.

## Table of Contents

- [Setup](#setup)
- [Audio Processing](#audio-processing)
- [Image Processing](#image-processing)
- [Video Processing](#video-processing)
- [Dependencies by Module](#dependencies-by-module)
- [Build & Run](#build--run)
- [Notes](#notes)
- [Changelog](#changelog)
- [License](#license)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/juliacwen/audio-image-video-toolkit-ai/
cd audio-image-video-toolkit-ai
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Repository Structure

```text
Audio/
  cpp/
  python/
Image/
  python/
Video/
  cpp/
  python/
```

(Full structure omitted for brevity; see individual folders for source and test files.)

## Audio Processing

Contains scripts and tools for audio processing: CSV generation, audio feature extraction, plotting, and AI-assisted workflows.

## Image Processing

The `Image/` directory contains Python scripts for detecting crescent moons in sky images. Multiple approaches are provided:

- **Classical Method**: `detect_crescent_classical.py`  
  Uses traditional HOG features and a linear SVM.

- **Machine Learning Methods**:  
  - `predict_crescent.py` — original ML-based SVM with augmentation  
  - `predict_crescent_pytorch.py` — PyTorch implementation with HOG features  
  - `predict_crescent_pytorch_cnn.py` — PyTorch CNN with augmentation, CPU/GPU switching, optional new-image temperature scaling  
  - `predict_crescent_tf.py` — TensorFlow CNN built with **tf.keras**, saves trained model as `.keras`  
  - `predict_crescent_tf_classic.py` — HOG + SVM classic approach, saves as `.h5`  
  - `predict_crescent_vit.py` — Vision Transformer–based model

- **Dataset generation**: `generate_crescent_images.py`  

## Video Processing

Contains C++ modules for motion estimation, video encoding, depth estimation, trajectory analysis, **plus Python scripts for video depth estimation**.

## Dependencies by Module

| Script / Tool | Domain | Libraries / Tools |
|---------------|--------|-----------------|
| Audio C++     | Audio  | g++, FFT, WAV parsing |
| Audio Python  | Audio  | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile |
| Audio AI Tools | Audio/AI | NumPy, Pandas, PyTorch |
| Image Python  | Image  | OpenCV, NumPy, scikit-learn, Matplotlib, joblib, tf.keras, PyTorch |
| Video Python  | Video  | OpenCV, PyTorch, timm, NumPy, Matplotlib |
| Video C++     | Video  | g++, standard C++ libraries, OpenCV |

## Build & Run

**C++ compilation** (Audio or Video tools):
```bash
make -C Audio/cpp
make -C Video/cpp/VideoEncodingAndVision
```

**Python scripts**
```bash
# Audio
cd Audio/python/src
python compare_csv.py

# Image
cd Image/python/src
python predict_crescent.py --image path/to/image.jpg
python predict_crescent_pytorch_cnn.py --image path/to/image.jpg
python predict_crescent_tf.py --image path/to/image.keras
python predict_crescent_tf_classic.py --image path/to/image.h5

# Video
cd Video/python/src
python video_depth_midas.py --video path/to/video.mp4
```

## Notes

- Some source code in this repository was **generated or assisted with AI tools**.
- AI assistance was used as a **development aid** only; core algorithmic logic and dataset-specific handling were implemented and verified manually.

## Changelog

- **2025‑09‑08** — Add image processing with PyTorch CNN model
- **2025‑09‑07** — Modernize Audio C++ 
- **2025‑09‑05** — Refactor and modernize Video C++ 
- **2025‑09‑04** — Added Video C++ modules for motion, encoding, and depth estimation; updated Video directory structure.
- **2025‑09‑02** — Added `predict_crescent_pytorch.py` and `predict_crescent_tf.py` to Image processing.
- **2025‑08‑29** — Added image crescent detection (`detect_crescent_classical.py`, `predict_crescent.py`), repo restructure.
- **2025‑08‑27** — Added AI FFT prediction workflow, Python/pytest validation.
- **2025‑08‑25** — Updated `wav_freq_csv` with FFT window support; AI-assisted windowing tests.
- **2025‑08‑23** — Added WAV → CSV, Python spectrum comparison, Python video pytest, WAV‑to‑WAV pytest.
- **2025‑08‑22** — Python audio compare: Torch overlay/diff with start/limit windowing.
- **2025‑08‑21** — Python video: MiDaS depth-estimation script.
- **2025‑08‑20** — C++ audio: WAV → CSV converter.
- **2025‑08‑18** — Initial setup.

## License

MIT License

