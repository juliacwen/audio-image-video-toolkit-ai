# Audio-Image-Video-Toolkit-AI
**Author:** Julia Wen (<wendigilane@gmail.com>)

## Project Overview
This repository contains projects for **Audio**, **Image**, and **Video** processing, organized into separate top-level directories. Each area contains source code (C++ and Python) and tests (Pytest and GoogleTest) intended for development and experimentation.

## Features

**Audio**: Converting WAV to CSV, generating FFT spectra with multiple window types, running end-to-end AI-assisted FFT workflows that generate test WAVs, train a small MLP on synthetic spectra, and predict tone probabilities, performing automated FFT windowing tests with rectangular, Hann, Hamming, and Blackman windows.

**Image**: Crescent detection in images, with multiple classical and machine learning approaches and dataset generation.

**Video**: C++ modules for motion estimation, video encoding, depth estimation, and trajectory analysis.

## Table of Contents

- [Setup](#setup)
- [Audio Processing](#audio-processing)
- [Image Processing](#image-processing)
- [Video Processing](#video-processing)
- [Dependencies by Module](#dependencies-by-module)
- [Build & Run](#build--run)
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
    src/
      wav_freq_csv.cpp
      wav_to_csv.cpp
    tests/
      test_wav_freq_csv.cpp
      test_wav_to_csv.cpp
      test_wav_freq_csv.py
      test_wav_to_csv.py
      test_ai_fft_windowing.py
      test_ai_fft_workflow.py
      ai_tools/
        ai_fft_windowing.py
        ai_test_all_windows.py
        ai_fft_workflow.py
        generate_wav.py
        nn_module.py
  python/
    src/
      compare_csv.py
      comparetorch_csv.py
      comp_plot_wav_diff.py
    tests/
      test_compare_csv.py
      test_comparetorch.py
      test_comp_plot_wav_diff.py

Image/
  python/
    src/
      detect_crescent_classical.py
      generate_crescent_images.py
      predict_crescent.py
      predict_crescent_pytorch.py
      predict_crescent_tf.py
      predict_crescent_vit.py
      dataset/
        crescent/
        no_crescent/

Video/
  cpp/
    VideoEncodingAndVision/
      src/
        video_common/
          inc/
            video_common.h
          src/
            video_common.cpp
        video_encoding/
          video_block_matching.cpp
          video_frame_prediction.cpp
          video_motion_estimation.cpp
          video_residual.cpp
        video_motion_and_depth/
          video_depth_from_stereo.cpp
          video_trajectory_from_motion.cpp
          video_vio_demo.cpp
      tests/
  python/
    src/
      video_depth_midas.py
    tests/
      test_video_depth_midas.py
```

## Audio Processing

Contains scripts and tools for audio processing: CSV generation, audio feature extraction, plotting, and AI-assisted workflows.

### Usage

**C++ Components**
- `wav_to_csv.cpp`: Converts WAV audio files to CSV.  
- `wav_freq_csv.cpp`: Extracts frequency-domain CSV.

**Python Components**
- `compare_csv.py`, `comp_plot_wav_diff.py`, `comparetorch_csv.py`

**AI Tools**
- `ai_fft_windowing.py`, `ai_fft_workflow.py`, `ai_test_all_windows.py`, `generate_wav.py`, `nn_module.py`

## Image Processing

The `Image/` directory contains Python scripts for detecting crescent moons in sky images. Multiple approaches are provided:

- **Classical Method**: `detect_crescent_classical.py`  
  Uses traditional HOG features and a linear SVM.

- **Machine Learning Methods**:  
  - `predict_crescent.py` — original ML-based SVM with augmentation  
  - `predict_crescent_pytorch.py` — PyTorch implementation  
  - `predict_crescent_tf.py` — TensorFlow implementation  
  - `predict_crescent_vit.py` — Vision Transformer–based model  

- **Dataset generation**: `generate_crescent_images.py`  

### Usage
```bash
cd Image/python/src

# Classical method
python detect_crescent_classical.py

# ML methods
python predict_crescent.py
python predict_crescent_pytorch.py
python predict_crescent_tf.py
python predict_crescent_vit.py
```

## Video Processing

Contains C++ modules for motion estimation, video encoding, depth estimation, and trajectory analysis.

### Notes

- `video_common`: Shared utilities for encoding and motion analysis  
- `video_encoding`: Block matching, frame prediction, motion estimation, residual computation  
- `video_motion_and_depth`: Stereo depth computation, motion-based trajectory estimation, VIO demo scripts  

### Usage

**C++ compilation**:
```bash
make -C Video/cpp/VideoEncodingAndVision
```

**Python scripts**:
```bash
cd Video/python/src
python video_depth_midas.py --video path/to/video.mp4
```

## Dependencies by Module

| Script / Tool | Domain | Libraries / Tools |
|---------------|--------|-----------------|
| Audio C++     | Audio  | g++, FFT, WAV parsing |
| Audio Python  | Audio  | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile |
| Audio AI Tools | Audio/AI | NumPy, Pandas, PyTorch |
| Image Python  | Image  | OpenCV, NumPy, scikit-learn, Matplotlib, joblib |
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

# Video
cd Video/python/src
python video_depth_midas.py --video path/to/video.mp4
```

## Changelog

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

