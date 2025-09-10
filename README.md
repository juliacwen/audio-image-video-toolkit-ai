# Audio-Image-Video-Toolkit-AI
**Author:** Julia Wen (<wendigilane@gmail.com>)

## Project Overview
This repository contains projects for **Audio**, **Image**, and **Video** processing, organized into separate top-level directories. Each area contains source code (C++ and Python) and tests (Pytest and GoogleTest) intended for development and experimentation with AI models.

## Features

**Audio**: Converting WAV to CSV, generating FFT spectra with multiple window types, running end-to-end AI-assisted FFT workflows that generate test WAVs, train a small MLP on synthetic spectra, and predict tone probabilities, performing automated FFT windowing tests with rectangular, Hann, Hamming, and Blackman windows. Supports **optional LLM explanations** if an OpenAI API key is set.

**Image**: Crescent detection in images, with multiple classical and machine learning approaches and dataset generation.

**Video**: C++ modules for motion estimation, video encoding, depth estimation, trajectory analysis, plus Python scripts for video depth estimation.

## Table of Contents

- [Setup](#setup)
- [Audio Processing](#audio-processing)
- [Image Processing](#image-processing)
- [Video Processing](#video-processing)
- [Optional LLM Explanation](#optional-llm-explanation)
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
    src/
    tests/
      ai_tools/
  python/
    src/
    tests/
Image/
  python/
    src/
      dataset/
Video/
  cpp/
    VideoEncodingAndVision/
      src/
      tests/
  python/
    src/
    tests/
```

## Audio Processing

Contains scripts and tools for audio processing: CSV generation, audio feature extraction, plotting, and AI-assisted workflows.

### C++ Components
- `Audio/cpp/src/wav_to_csv.cpp`: WAV → CSV (supports PCM **16-bit**, **24-bit**, and **IEEE Float32**; outputs `Index,Sample`).  
    *Implementation: manual WAV parsing in C++.*
- `Audio/cpp/src/wav_freq_csv.cpp`: WAV → CSV **and** FFT Spectrum CSV (`Index,Sample` and `Frequency,Magnitude`).  
    *Implementation: manual WAV parsing + FFT using `std::complex`. Supports selectable FFT windows: rectangular, Hann, Hamming, Blackman.*

- `Audio/cpp/test/test_wav_to_csv.cpp`: googletest
- `Audio/cpp/test/test_wav_freq_csv.cpp`: googletest

### Python Components
- `Audio/python/src/compare_csv.py` — compare two **time-domain WAV CSVs**.  
    *Uses **NumPy**, **Pandas**, **Matplotlib**.*
- `Audio/python/src/comparetorch_csv.py` — Uses PyTorch tensors for faster handling of large CSVs. compare **time-domain or spectrum CSVs** with overlay + diff.  
- `Audio/python/src/comp_plot_wav_diff.py` — compare **two WAV audio files** directly.  
    *Uses **torchaudio**, **PyTorch**, **Matplotlib**.*

### Pytest with AI-assisted tools
- `Audio/cpp/tests/test_wav_to_csv.py` — pytest
- `Audio/cpp/tests/test_wav_freq_csv.py` — pytest
- `Audio/cpp/tests/test_ai_fft_windowing.py` — executes automated tests
- `Audio/cpp/tests/test_ai_fft_workflow.py` — executes automated AI workflow tests
- `Audio/cpp/tests/ai_tools/ai_fft_windowing.py` — Apply different FFT windowing functions (rectangular, Hann, Hamming, Blackman) to audio segments.
- `Audio/cpp/tests/ai_tools/ai_fft_workflow.py` — AI-assisted FFT workflow: generate WAVs, compute FFT, optional LLM explanation.
- `Audio/cpp/tests/ai_tools/ai_test_all_windows.py` — Automated testing for all window types.
- `Audio/cpp/tests/ai_tools/generate_wav.py` — Generate synthetic WAV files.
- `Audio/cpp/tests/ai_tools/nn_module.py` — Small MLP model for tone prediction.

### End-to-end AI FFT Test Workflow
- `Audio/python/tests/ai_llm_fft_demo.py` — AI FFT demo with optional LLM explanation.
- `Audio/python/tests/test_comp_plot_wav_diff.py` - pytest for comp_plot_wav_diff.py
- `Audio/python/tests/test_compare_csv.py` - pytest for compare_csv.py
- `Audio/python/tests/test_comparetorch.py` - pytest for comparetorch_csv.py

## Image Processing

Python scripts for detecting crescent moons in sky images.

**Detection approaches:**
- `predict_crescent.py` – Python SVM with HOG + augmentation  
- `predict_crescent_pytorch.py` – PyTorch-based SVM classifier using HOG features  
- `predict_crescent_pytorch_cnn.py` – PyTorch CNN classifier with augmentation, CPU/GPU switching, and optional temperature scaling for new images  
- `predict_crescent_tf.py` – TensorFlow CNN classifier built with tf.keras; saves trained model as a `.keras` file  
- `predict_crescent_tf_classic.py` – TensorFlow wrapper around classic HOG + SVM approach; saves model as a `.h5` file  
- `detect_crescent_classical.py` – Classical HOG + SVM approach  
- `predict_crescent_vit.py` – Vision Transformer-based crescent prediction

**Dataset generation:** `generate_crescent_images.py`

### Usage
```bash
cd Image/python/src
python detect_crescent_classical.py
python predict_crescent.py
python predict_crescent_pytorch.py
python predict_crescent_pytorch_cnn.py
python predict_crescent_tf.py
python predict_crescent_tf_classic.py
python predict_crescent_vit.py
```

## Video Processing

C++ modules for motion estimation, video encoding, depth estimation, trajectory analysis, plus Python scripts for depth estimation.

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

## Optional LLM Explanation

Example Scripts (e.g., `ai_llm_fft_demo.py`) to generate **plain-language explanations of FFT results** using OpenAI models.

### Requirements
```bash
export OPENAI_API_KEY="your_api_key_here"   # macOS/Linux
setx OPENAI_API_KEY "your_api_key_here"     # Windows
export LLM_MODEL="gpt-4o-mini"  # optional, default: gpt-3.5-turbo
```

**Behavior**
- If the API key exists, top 5 average frequency magnitudes per window type are sent to the LLM.
- The LLM returns a **natural-language explanation**.
- If no API key is set, offline FFT analysis and plotting still function normally.

## Dependencies by Module

| Script / Tool   | Domain      | Libraries / Tools |
|-----------------|------------|-----------------|
| Audio C++       | Audio      | g++, standard C++ libraries, FFT, WAV parsing |
| Audio Python    | Audio      | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile, pandas, openpyxl, python-dotenv |
| Audio AI Tools  | Audio/AI   | NumPy, Pandas, PyTorch, openpyxl, python-dotenv, (optional: LangChain, OpenAI Python SDK) |
| Image Python    | Image      | OpenCV, NumPy, scikit-learn, Matplotlib, joblib, tf.keras, PyTorch |
| Video C++       | Video      | g++, standard C++ libraries, OpenCV |
| Video Python    | Video      | OpenCV, PyTorch, timm, NumPy, Matplotlib |

## Build & Run

**C++ compilation**
```bash
make -C Audio/cpp
make -C Video/cpp/VideoEncodingAndVision
```
**Python scripts**
```bash
# Audio
cd Audio/python/src
python compare_csv.py
python ai_fft_demo.py ../../test_files/readme_sample.wav

# Image
cd Image/python/src
python predict_crescent.py --image path/to/image.jpg

# Video
cd Video/python/src
python video_depth_midas.py --video path/to/video.mp4
```

## Notes

- Some source code was **assisted with AI tools**, but core logic was implemented and verified manually.
- The FFT AI demo is **offline-first**; LLM explanation is optional.

## Changelog
- **2025‑09‑10** — Added AI FFT flow with LLM demo
- **2025‑09‑08** — Added image processing with PyTorch CNN model 
- **2025‑09‑07** — Modernize Audio C++ 
- **2025‑09‑05** — Refactor Video C++  
- **2025‑09‑04** — Added Video C++ modules for motion, encoding, depth estimation; updated structure  
- **2025‑09‑02** — Added `predict_crescent_pytorch.py` and `predict_crescent_tf.py`  
- **2025‑08‑29** — Added crescent detection scripts; repo restructure  
- **2025‑08‑27** — Added AI FFT prediction workflow, Python/pytest validation  
- **2025‑08‑25** — Updated `wav_freq_csv` with FFT window support; AI-assisted tests  
- **2025‑08‑23** — Added WAV→CSV, spectrum comparison, video pytest  
- **2025‑08‑22** — Python audio compare: Torch overlay/diff with start/limit windowing  
- **2025‑08‑21** — Python video: MiDaS depth-estimation script  
- **2025‑08‑20** — C++ audio: WAV → CSV converter  
- **2025‑08‑18** — Initial setup

## License
MIT License

