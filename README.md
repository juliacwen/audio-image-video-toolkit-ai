# Audio-Image-Video-Toolkit-AI
**Author:** Julia Wen (<wendigilane@gmail.com>)
# Project Overview
This repository contains projects for **Audio**, **Image**, and **Video** processing, organized into separate top-level directories. Each area contains source code (C++ and Python) and tests (Pytest and GoogleTest) intended for development and experimentation.

Features: 
Audio: Converting WAV to CSV, generating FFT spectra with multiple window types, running end-to-end AI-assisted FFT workflows that generate test WAVs, train a small MLP on synthetic spectra, and predict tone probabilities, performing automated FFT windowing tests with rectangular, Hann, Hamming, and Blackman windows.
Image: Crescent detection in images:
- `detect_crescent_classical.py`: Uses classical HOG features and a linear SVM for crescent detection.
- `predict_crescent.py`: Uses machine learning with data augmentation, SVM, and optional image display for improved prediction and probability estimates.

Video: Running MiDaS depth estimation on videos

## Table of Contents

- [Setup](#setup)  
- [Audio Processing](#audio-processing)  
- [Image Processing](#image-processing)  
- [Video Processing](#video-processing)  
- [Dependencies by Module](#dependencies-by-module)  
- [Build](#build)  
- [Run](#run)  

```

## Setup
### 1. Clone the repository
```bash
git clone https://github.com/juliacwen/audio-image-video-toolkit-ai/
cd <your-repo>
```
### 2. Create a virtual environment
```bash
python3 -m venv venv
```
### 3. Activate the virtual environment (single code block)
```bash
# On macOS / Linux
source venv/bin/activate
# On Windows (PowerShell)
venv\Scripts\activate
```
### 4. Install dependencies

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
  python/
    tests/

## Audio Processing  

The `Audio/` directory contains scripts and tools for audio processing, including CSV generation, audio feature extraction, plotting, and machine learning–based workflows. Both C++ and Python implementations are provided.

### Directory Structure

```bash
Audio/
├── cpp/
│   ├── src/
│   │   ├── wav_freq_csv.cpp
│   │   └── wav_to_csv.cpp
│   └── tests/
│       ├── test_wav_freq_csv.cpp
│       ├── test_wav_to_csv.cpp
│       ├── test_wav_freq_csv.py
│       ├── test_wav_to_csv.py
│       ├── test_ai_fft_windowing.py
│       ├── test_ai_fft_workflow.py
│       └── ai_tools/
│           ├── ai_fft_windowing.py
│           ├── ai_test_all_windows.py
│           ├── ai_fft_workflow.py
│           ├── generate_wav.py
│           └── nn_module.py
└── python/
    ├── src/
    │   ├── compare_csv.py
    │   ├── comparetorch_csv.py
    │   └── comp_plot_wav_diff.py
    └── tests/
        ├── test_compare_csv.py
        ├── test_comparetorch.py
        └── test_comp_plot_wav_diff.py


### Requirements

The audio scripts depend on:  
- **NumPy, SciPy** – numerical and signal processing  
- **Matplotlib** – waveform and frequency-domain plotting  
- **scikit-learn** – machine learning models for audio analysis  
- **SoundFile** – reading and writing audio files  
- **PyTorch** – for neural network–based experiments  

### Usage
### C++ Components

- **`wav_to_csv.cpp`**: Converts raw WAV audio files into CSV format for further analysis.  
- **`wav_freq_csv.cpp`**: Extracts frequency-domain data from WAV files and saves it as CSV.  

#### C++ Tests
- **`test_wav_to_csv.cpp`** and **`test_wav_freq_csv.cpp`** validate WAV-to-CSV conversion.  
- Python-based test wrappers (**`test_wav_to_csv.py`**, **`test_wav_freq_csv.py`**) check integration with the workflow.  

### Python Components

- **`compare_csv.py`**: Compares two CSV files for differences in audio features.  
- **`comp_plot_wav_diff.py`**: Plots differences between waveforms extracted from audio files.  
- **`comparetorch_csv.py`**: Extends comparison using PyTorch for additional analysis.  

#### Python Tests
- **`test_compare_csv.py`**, **`test_comp_plot_wav_diff.py`**, and **`test_comparetorch.py`** validate the Python audio functionality.  

### AI Tools (in C++ Tests)

Located in `Audio/cpp/tests/ai_tools/`:  

- **`ai_fft_windowing.py`**: Implements FFT windowing for audio feature extraction.  
- **`ai_test_all_windows.py`**: Tests different FFT window functions for accuracy.  
- **`ai_fft_workflow.py`**: Defines a complete pipeline for FFT-based audio analysis.  
- **`generate_wav.py`**: Script to generate WAV files for testing.  
- **`nn_module.py`**: Defines neural network modules for audio-related ML tasks.  

## Image Processing  

The `Image/` directory contains Python scripts for detecting crescent moons in sky images. Two approaches are provided:  

- **Classical Method (`detect_crescent_classical.py`)**: Uses traditional image processing techniques.  
- **Machine Learning Method (`predict_crescent.py`)**: Uses an SVM model trained on labeled crescent vs. non-crescent images. The trained model is saved as `crescent_moon_model.pkl`.  

### Directory Structure
```bash
Image/
└── python/
    └── src/
        ├── detect_crescent_classical.py
        ├── predict_crescent.py
        ├── README.md
        └── dataset/
            ├── crescent/
            └── no_crescent/

### Requirements
opencv-python   # image loading and preprocessing
numpy           # array manipulation
scikit-learn    # SVM classifier for crescent prediction
matplotlib      # plotting and visualization

### Usage
cd Image/python/src

# Classical method (predicts on dataset images)
python detect_crescent_classical.py

# Machine learning method (train/predict; can pass --image to predict a single external image)
python predict_crescent.py
python predict_crescent.py --image path/to/image.jpg

## Video Processing

The `Video/` directory contains Python scripts for video processing tasks. Currently, it includes depth estimation from video frames using a pre-trained model.

### Directory Structure
```bash
Video/
└── python/
    ├── src/
    │   └── video_depth_midas.py
    └── tests/
        └── test_video_depth_midas.py

### Requirements
opencv-python   # video reading and processing
torch            # PyTorch for model inference
timm             # pre-trained model support
numpy            # array manipulation
matplotlib       # optional plotting

### Usage
cd Video/python/src

# Run depth estimation on a video
python video_depth_midas.py --video path/to/video.mp4


## Dependencies by Module

| Tool/Script                                | Domain        | Libraries / Tools Used                              |
|--------------------------------------------|---------------|-----------------------------------------------------|
| `Audio/cpp/src/wav_to_csv.cpp`             | Audio (C++)   | g++, C++ standard library (manual WAV parse)        |
| `Audio/cpp/src/wav_freq_csv.cpp`           | Audio (C++)   | g++, C++ standard library (`std::complex` FFT)      |
| `Audio/python/src/compare_csv.py`          | Audio (Py)    | NumPy, Pandas, Matplotlib                           |
| `Audio/python/src/comparetorch_csv.py`     | Audio (Py)    | PyTorch, Matplotlib                                 |
| `Audio/python/src/comp_plot_wav_diff.py`   | Audio (Py)    | torchaudio, PyTorch, Matplotlib                     |
| `Audio/cpp/tests/ai_tools/ai_fft_windowing.py` | Audio/AI  | NumPy, SciPy, Python stdlib (subprocess, math)      |
| `Audio/cpp/tests/ai_tools/ai_fft_workflow.py` | Audio/AI  | NumPy, Pandas, PyTorch                              |
| `Audio/cpp/tests/ai_tools/nn_module.py`    | Audio/AI      | PyTorch                                             |
| `Image/python/src/detect_crescent_classical.py` | Image (Py)| OpenCV, NumPy                                       |
| `Image/python/src/predict_crescent.py`     | Image (Py)    | scikit-learn (SVM), NumPy, Matplotlib, joblib       |
| `Video/python/src/video_depth_midas.py`    | Video (Py)    | OpenCV, PyTorch (MiDaS models), NumPy               |


## Notes

- Python scripts use standard scientific libraries: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `torch`, `timm`, `opencv-python`, `imageio`.
- Audio C++ scripts depend on standard compilation and linking with appropriate libraries (no pre-trained models required).
- Test scripts are included for both Python and C++ components to validate functionality.
- The trained SVM model for crescent detection is saved as `crescent_moon_model.pkl`.
- The `predict_crescent.py` script can take external images for prediction using the `--image` flag.
- Image augmentation is applied during training to improve model robustness.
- The audio scripts generate CSVs, perform FFT-based analysis, and include AI-based audio processing tools.
- Video scripts perform depth estimation from video frames using pre-trained models.

## Notes
- WAV reader supports PCM **16‑bit**, **24‑bit**, and **Float32**.  
- CSV formats:
  - WAV CSV: `Index,Sample`
  - Spectrum CSV: `Frequency,Magnitude`
- Stereo WAVs are averaged to mono.  
- FFT windowing tests generate WAVs and validate multiple FFT window types.  
- AI workflow tests generate WAVs, train MLP, predict tone probabilities, write `predictions.txt`.

## Changelog
- **2025‑08‑29** — Added image crescent detection with two models: machine learning vs classic model and repo lay restrtucture.
- **2025‑08‑27** — Added end-to-end AI FFT prediction test workflow; Python/pytest validation (`ai_fft_workflow.py` + `nn_module.py` + `test_ai_fft_workflow.py`).  
- **2025‑08‑25** — Updated `wav_freq_csv` with FFT window support; added AI-assisted FFT windowing tests; Python/pytest wrapper for all windows, 1 kHz sine at 8 kHz sample rate.  
- **2025‑08‑23** — Added `wav_freq_csv.cpp` (WAV → CSV + FFT spectrum), Python spectrum comparison in `comparetorch_csv.py`, Python video pytest (`test_video_depth_midas.py`), and WAV‑to‑WAV pytest (`test_comp_plot_wav_diff.py`).  
- **2025‑08‑22** — Python audio compare: Torch overlay/diff with start/limit windowing.  
- **2025‑08‑21** — Python video: MiDaS depth‑estimation script.  
- **2025‑08‑20** — C++ audio: WAV → CSV converter.  
- **2025‑08‑18** — Initial setup.

## License

This project is released under the MIT License.



## Table of Contents
- [Features](#features)
- [Dependencies by Module](#dependencies-by-module)
- [Project Layout](#project-layout)
- [Build (C++)](#build-c)
- [C++ Tests](#c-tests)
- [Python Tools](#python-tools)
- [Python Tests](#python-tests)
- [Notes](#notes)
- [Changelog](#changelog)
- [License](#license)

## Features

- **C++ Tools**
  - `wav_to_csv.cpp`: WAV → CSV (supports PCM **16-bit**, **24-bit**, and **IEEE Float32**; outputs `Index,Sample`).  
    *Implementation: manual WAV parsing in C++.*
  - `wav_freq_csv.cpp`: WAV → CSV **and** FFT Spectrum CSV (`Index,Sample` and `Frequency,Magnitude`).  
    *Implementation: manual WAV parsing + FFT using `std::complex`. Supports selectable FFT windows: rectangular, Hann, Hamming, Blackman.*

- **Python Audio Tools**
  - `python/audio/compare_csv.py` — compare two **time-domain WAV CSVs**.  
    *Uses **NumPy**, **Pandas**, **Matplotlib**.*
  - `python/audio/comparetorch_csv.py` — compare **time-domain or spectrum CSVs** with overlay + diff.  
    *Uses **PyTorch** for tensor ops, **Matplotlib** for plotting.*
  - `python/audio/comp_plot_wav_diff.py` — compare **two WAV audio files** directly.  
    *Uses **torchaudio**, **PyTorch**, **Matplotlib**.*

- **Python Video Tools**
  - `python/video/video_depth_midas.py` — MiDaS depth-estimation to MP4 with selectable models.  
    **Flags**
    - `--model` — MiDaS variant (`DPT_Hybrid`, `DPT_Large`, `MiDaS_small`)  
    - `--debug` — verbose logging and optional intermediate outputs  

- **AI-assisted / Automated FFT Windowing Tests**
  - Updated `wav_freq_csv` to support **multiple FFT windows**.
  - Python/pytest wrapper `cpp/tests/test_ai_fft_windowing.py` executes automated tests:
    - Generates **1 kHz sine WAV at 8 kHz sample rate**.
    - Tests FFT with **rectangular, Hann, Hamming, and Blackman** windows.
    - Produces CSV outputs for validation and comparison.
  - `cpp/tests/ai_tools/ai_fft_windowing.py` — helper for windowing operations in these tests.

- **End-to-end AI FFT Prediction Test Workflow**
  - Python workflow for **testing AI predictions on spectra**:
    - `cpp/tests/ai_tools/ai_fft_workflow.py` — generates WAVs, converts WAV → CSV + FFT spectrum, trains a small MLP (`nn_module.py`), predicts tone presence, writes probabilities to `predictions.txt`.
    - `cpp/tests/ai_tools/nn_module.py` — defines the MLP used by `ai_fft_workflow.py`.
    - `cpp/tests/test_ai_fft_workflow.py` — pytest validating the AI workflow end-to-end.

## Dependencies by Module

| Tool/Script                      | Domain        | Libraries Used                           |
|----------------------------------|---------------|------------------------------------------|
| `cpp/audio/wav_to_csv.cpp`       | Audio (C++)   | C++ standard library (manual WAV parse)  |
| `cpp/audio/wav_freq_csv.cpp`     | Audio (C++)   | C++ standard library (`std::complex` FFT)|
| `python/audio/compare_csv.py`    | Audio (Py)    | NumPy, Pandas, Matplotlib                |
| `python/audio/comparetorch_csv.py` | Audio (Py)  | PyTorch, Matplotlib                      |
| `python/audio/comp_plot_wav_diff.py` | Audio (Py)| torchaudio, PyTorch, Matplotlib          |
| `python/video/video_depth_midas.py` | Video (Py) | OpenCV, PyTorch (MiDaS models)           |
| `cpp/tests/ai_tools/ai_fft_windowing.py` | Audio/AI | Python standard library (subprocess, math) |
| `cpp/tests/ai_tools/ai_fft_workflow.py` | Audio/AI | NumPy, Pandas, PyTorch                  |
| `cpp/tests/ai_tools/nn_module.py` | Audio/AI | PyTorch                                 |

## Project Layout
```text
cpp/
  audio/
    wav_to_csv.cpp
    wav_freq_csv.cpp
  tests/
    test_wav_to_csv.cpp
    test_wav_freq_csv.cpp
    test_wav_to_csv.py
    test_wav_freq_csv.py
    test_ai_fft_windowing.py
    ai_tools/
      ai_test_all_windows.py
      generate_wav.py
      ai_fft_workflow.py
      nn_module.py
      ai_fft_windowing.py
python/
  audio/
    compare_csv.py
    comparetorch_csv.py
    comp_plot_wav_diff.py
  video/
    video_depth_midas.py
  tests/
    test_compare_csv.py
    test_comparetorch.py
    test_comp_plot_wav_diff.py
    test_video_depth_midas.py
    test_ai_fft_workflow.py

## Build (C++)
Build with the Makefile:
```bash
make all
```
This compiles:
- `wav_to_csv`
- `wav_freq_csv`
- `test_wav_to_csv`
- `test_wav_freq_csv`

Binaries are placed in the project root (`./wav_to_csv`, `./wav_freq_csv`).

### (Optional) Manual compile
```bash
# Tools
g++ -std=c++17 -O2 -o wav_to_csv cpp/audio/wav_to_csv.cpp -lm
g++ -std=c++17 -O2 -o wav_freq_csv cpp/audio/wav_freq_csv.cpp -lm

# C++ tests
g++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib   cpp/tests/test_wav_to_csv.cpp -lgtest -lgtest_main -pthread -lm -o test_wav_to_csv

g++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib   cpp/tests/test_wav_freq_csv.cpp -lgtest -lgtest_main -pthread -lm -o test_wav_freq_csv
```

## C++ Tests

### Run C++ unit tests (gtest)
```bash
make test
```
Runs:
- `test_wav_to_csv`
- `test_wav_freq_csv`
- `test_ai_fft_windowing` (via pytest wrapper)

### Run pytest wrappers for C++ tools
```bash
pytest cpp/tests
```
Invokes compiled binaries and AI/automated FFT windowing tests, validates generated CSV output.

---

## Python Tools

### 1) Compare WAV CSVs
```bash
python3 python/audio/compare_csv.py fileA.csv fileB.csv
```

### 2) Compare CSVs with Torch
```bash
python3 python/audio/comparetorch_csv.py fileA.csv fileB.csv [start] [limit]
```

### 3) Compare two WAV files directly
```bash
python3 python/audio/comp_plot_wav_diff.py fileA.wav fileB.wav [--zoom Z] [--save out.png]
```

### 4) Video depth (MiDaS)
```bash
python3 python/video/video_depth_midas.py sample.mp4 out_depth.mp4 --model DPT_Hybrid --debug
```

---

## Python Tests
Run all tests:
```bash
pytest python/tests
```
Run all C++ tests including AI/automated FFT windowing:
```bash
pytest cpp/tests
```

---

## Notes
- WAV reader supports PCM **16‑bit**, **24‑bit**, and **Float32**.  
- CSV formats:
  - WAV CSV: `Index,Sample`
  - Spectrum CSV: `Frequency,Magnitude`
- Stereo WAVs are averaged to mono.  
- FFT windowing tests generate WAVs and validate multiple FFT window types.  
- AI workflow tests generate WAVs, train MLP, predict tone probabilities, write `predictions.txt`.

## Changelog
- **2025‑08‑27** — Added end-to-end AI FFT prediction test workflow; Python/pytest validation (`ai_fft_workflow.py` + `nn_module.py` + `test_ai_fft_workflow.py`).  
- **2025‑08‑25** — Updated `wav_freq_csv` with FFT window support; added AI-assisted FFT windowing tests; Python/pytest wrapper for all windows, 1 kHz sine at 8 kHz sample rate.  
- **2025‑08‑23** — Added `wav_freq_csv.cpp` (WAV → CSV + FFT spectrum), Python spectrum comparison in `comparetorch_csv.py`, Python video pytest (`test_video_depth_midas.py`), and WAV‑to‑WAV pytest (`test_comp_plot_wav_diff.py`).  
- **2025‑08‑22** — Python audio compare: Torch overlay/diff with start/limit windowing.  
- **2025‑08‑21** — Python video: MiDaS depth‑estimation script.  
- **2025‑08‑20** — C++ audio: WAV → CSV converter.  
- **2025‑08‑18** — Initial setup.

## License
MIT License
