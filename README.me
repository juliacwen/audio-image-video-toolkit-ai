# Audio & Video Tools

**Author:** Julia Wen (<wendigilane@gmail.com>)

Utilities for working with audio and video data: converting WAV to CSV, generating FFT spectrum, comparing waveforms/spectrums, running AI-assisted FFT windowing tests, and running MiDaS depth on videos.

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
    *Uses **torchaudio** (for WAV I/O), **PyTorch**, **Matplotlib**.*

- **Python Video Tools**
  - `python/video/video_depth_midas.py` — MiDaS depth-estimation to MP4 with selectable models.  
    **Flags**
    - `--model` — MiDaS variant:
      - `DPT_Hybrid` — balance of quality/speed (default recommended).
      - `DPT_Large` — higher quality, slower.
      - `MiDaS_small` — fastest, lower quality.
    - `--debug` — verbose logging and optional intermediate outputs.  
    **Backends:** OpenCV, PyTorch.  
    **Note:** MiDaS = Mixed-Domain Attention Stereo depth estimation model (Intel/ETH Zürich, PyTorch implementation).

- **AI-assisted / Automated FFT Windowing**
  - Updated `wav_freq_csv` to support **multiple FFT windows**.
  - Python/pytest wrapper `cpp/tests/test_ai_fft_windowing.py` executes `ai_test_all_windows.py`:
    - Generates **1 kHz sine WAV at 8 kHz sample rate**.
    - Tests FFT with **rectangular, Hann, Hamming, and Blackman** windows.
    - Produces CSV outputs for validation and comparison.

## Dependencies by Module

| Tool/Script                      | Domain        | Libraries Used                           |
|----------------------------------|---------------|------------------------------------------|
| `cpp/audio/wav_to_csv.cpp`       | Audio (C++)   | C++ standard library (manual WAV parse)  |
| `cpp/audio/wav_freq_csv.cpp`     | Audio (C++)   | C++ standard library (`std::complex` FFT)|
| `python/audio/compare_csv.py`    | Audio (Py)    | NumPy, Pandas, Matplotlib                |
| `python/audio/comparetorch_csv.py` | Audio (Py)  | PyTorch, Matplotlib                      |
| `python/audio/comp_plot_wav_diff.py` | Audio (Py)| torchaudio, PyTorch, Matplotlib          |
| `python/video/video_depth_midas.py` | Video (Py) | OpenCV, PyTorch (MiDaS models)           |
| `cpp/tests/ai_tools/ai_test_all_windows.py` | Audio/AI | Python standard library (subprocess, math) |

## Project Layout
```
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
```

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
- C++ WAV reader supports PCM **16‑bit**, **24‑bit**, and **Float32**.  
- CSV formats:
  - WAV CSV: `Index,Sample`
  - Spectrum CSV: `Frequency,Magnitude`
- Stereo WAVs are averaged to mono.  
- AI/automated FFT windowing tests generate test WAVs and compare multiple FFT windows.  
- On macOS with Homebrew, gtest is commonly under `/opt/homebrew/include` and `/opt/homebrew/lib`.

## Changelog
- **2025‑08‑25** — Updated `wav_freq_csv` with FFT window support; **added AI-assisted FFT windowing tests**; Python/pytest wrapper for all windows, 1 kHz sine at 8 kHz sample rate; updated README to reflect changes.  
- **2025‑08‑23** — Added `wav_freq_csv.cpp` (WAV → CSV + FFT spectrum), Python spectrum comparison in `comparetorch_csv.py`, Python video pytest (`test_video_depth_midas.py`), and WAV‑to‑WAV pytest (`test_comp_plot_wav_diff.py`).  
- **2025‑08‑22** — Python audio compare: Torch overlay/diff with start/limit windowing.  
- **2025‑08‑21** — Python video: MiDaS depth‑estimation script.  
- **2025‑08‑20** — C++ audio: WAV → CSV converter.  
- **2025‑08‑18** — Initial setup.


## License
MIT License
