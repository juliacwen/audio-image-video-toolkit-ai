# Audio & Video Tools

**Author:** Julia Wen (<wendigilane@gmail.com>)

Utilities for working with audio and video data: converting WAV to CSV, generating FFT spectrum, comparing waveforms/spectrums, and running MiDaS depth on videos.

## Table of Contents
- [Features](#features)
- [**Dependencies by Module**](#dependencies-by-module)
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
  - `wav_freq_csv.cpp`: WAV → CSV **and** FFT Spectrum CSV (`Index,Sample` and `Frequency,Magnitude`) with optional windowing (hann, hamming, blackman, rectangular).  
    *Implementation: manual WAV parsing + FFT using `std::complex`.*

- **Python Audio Tools**
  - `python/audio/compare_csv.py` — compare two **time-domain WAV CSVs**.  
    *Uses **NumPy**, **Pandas**, **Matplotlib** (no Torch).*
  - `python/audio/comparetorch_csv.py` — compare **time-domain or spectrum CSVs** with overlay + diff.  
    *Uses **PyTorch** for tensor ops, **Matplotlib** for plotting.*
  - `python/audio/comp_plot_wav_diff.py` — compare **two WAV audio files** directly (PCM 16/24-bit or Float32).  
    *Uses **torchaudio** (for WAV I/O), **PyTorch** (ops), **Matplotlib** (plots).*

- **Python Video Tools**
  - `python/video/video_depth_midas.py` — MiDaS (Mixed-Domain Attention Stereo) depth-estimation to MP4 with selectable models.  
    *Uses **OpenCV** (video I/O), **PyTorch** (MiDaS models).*

- **Tests**
  - **C++**: GoogleTest (`cpp/tests/test_wav_to_csv.cpp`, `cpp/tests/test_wav_freq_csv.cpp`) and pytest wrappers for binaries (`cpp/tests/test_wav_to_csv.py`, `cpp/tests/test_wav_freq_csv.py`); test_wav_freq_csv tests all windows (hann, hamming, blackman, rectangular) and 1 kHz sine at 8 kHz.
  - **Python**: pytest for audio & video tools (`python/tests/`).

## **Dependencies by Module**

| Tool/Script                      | Domain        | Libraries Used                           |
|----------------------------------|---------------|------------------------------------------|
| `cpp/audio/wav_to_csv.cpp`       | Audio (C++)   | C++ standard library (manual WAV parse)  |
| `cpp/audio/wav_freq_csv.cpp`     | Audio (C++)   | C++ standard library (`std::complex` FFT)|
| `python/audio/compare_csv.py`    | Audio (Py)    | NumPy, Pandas, Matplotlib                |
| `python/audio/comparetorch_csv.py` | Audio (Py)  | PyTorch, Matplotlib                      |
| `python/audio/comp_plot_wav_diff.py` | Audio (Py)| torchaudio, PyTorch, Matplotlib          |
| `python/video/video_depth_midas.py` | Video (Py) | OpenCV, PyTorch (MiDaS models)           |

## Project Layout
```
cpp/
  audio/
    wav_to_csv.cpp
    wav_freq_csv.cpp
  tests/
    test_wav_to_csv.cpp        # gtest for wav_to_csv
    test_wav_freq_csv.cpp      # gtest for wav_freq_csv
    test_wav_to_csv.py         # pytest wrapper for wav_to_csv (calls compiled binary)
    test_wav_freq_csv.py       # pytest wrapper for wav_freq_csv (calls compiled binary)

python/
  audio/
    compare_csv.py             # NumPy/Pandas
    comparetorch_csv.py        # Torch
    comp_plot_wav_diff.py      # torchaudio + Torch
  video/
    video_depth_midas.py       # OpenCV + Torch (MiDaS)
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

> Binaries are placed in the project root (`./wav_to_csv`, `./wav_freq_csv`).

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
Runs both:
- `test_wav_to_csv`
- `test_wav_freq_csv`

### Run pytest wrappers for C++ tools
```bash
pytest cpp/tests
```
These Python tests invoke the compiled binaries (`./wav_to_csv`, `./wav_freq_csv`) and validate generated CSV output.

---

## Python Tools

### 1) Compare WAV CSVs (time‑domain only)
Script: `python/audio/compare_csv.py`  
Usage:
```bash
python3 python/audio/compare_csv.py fileA.csv fileB.csv
```
- Overlays two time‑domain WAV CSVs (`Index,Sample`) and shows difference.  
- **Backends:** NumPy, Pandas, Matplotlib.

### 2) Compare CSVs with Torch (time‑domain or spectrum)
Script: `python/audio/comparetorch_csv.py`  
Usage:
```bash
python3 python/audio/comparetorch_csv.py fileA.csv fileB.csv [start] [limit]
```
- Accepts **time‑domain** WAV CSVs or **spectrum** CSVs.  
- Visualization: overlay (blue solid vs orange dashed) + difference (red).  
- Optional `start` & `limit` to window the comparison.  
- **Backends:** PyTorch, Matplotlib.

### 3) Compare two WAV files (direct)
Script: `python/audio/comp_plot_wav_diff.py`  
Usage:
```bash
python3 python/audio/comp_plot_wav_diff.py fileA.wav fileB.wav [--zoom Z] [--save out.png]
```
- Loads audio via torchaudio and compares waveforms directly.  
- Supports PCM **16-bit**, **24-bit**, and **Float32** WAVs.  
- `--zoom` to focus on a small segment; `--save` to write plot without GUI.  
- **Backends:** torchaudio, PyTorch, Matplotlib.

### 4) Video depth (MiDaS)
Script: `python/video/video_depth_midas.py`  
Usage:
```bash
python3 python/video/video_depth_midas.py sample.mp4 out_depth.mp4 --model DPT_Hybrid --debug
```
**Flags**
- `--model` — MiDaS variant:
  - `DPT_Hybrid` — balance of quality/speed (default recommended).
  - `DPT_Large` — higher quality, slower.
  - `MiDaS_small` — fastest, lower quality.
- `--debug` — verbose logging and optional intermediate outputs.  
- **Backends:** OpenCV, PyTorch.  
- **Note:** *MiDaS = Mixed-Domain Attention Stereo depth estimation model (Intel/ETH Zürich, PyTorch implementation).*

---

## Python Tests

Run **all Python tests** (audio + video):
```bash
pytest python/tests
```
Run an **individual** test:
```bash
pytest python/tests/test_comp_plot_wav_diff.py
```

Included tests:
- `test_compare_csv.py` — time‑domain CSV overlay/diff (**NumPy/Pandas**).  
- `test_comparetorch.py` — time‑domain & spectrum CSV overlay/diff (**Torch**).  
- `test_comp_plot_wav_diff.py` — validates WAV‑to‑WAV comparison (**torchaudio**, Torch).  
- `test_video_depth_midas.py` — dummy MP4 → depth MP4 (**OpenCV**, Torch, MiDaS).

---

## Notes
- C++ WAV reader supports PCM **16‑bit**, **24‑bit**, and **Float32**.  
- CSV formats:
  - WAV CSV: `Index,Sample`
  - Spectrum CSV: `Frequency,Magnitude`
- Stereo WAVs are averaged to mono.  
- On macOS with Homebrew, gtest is commonly under `/opt/homebrew/include` and `/opt/homebrew/lib`.

## Changelog
- **2025-08-25** — Updated wav_freq_csv with window support; Python/pytest wrapper for all windows, 1 kHz sine at 8 kHz
- **2025‑08‑23** — Added `wav_freq_csv.cpp` (WAV → CSV + FFT spectrum), Python spectrum comparison in `comparetorch_csv.py`, **Python video pytest** (`test_video_depth_midas.py`), and **WAV‑to‑WAV pytest** (`test_comp_plot_wav_diff.py`); dedicated tests (C++ gtest + pytest), Makefile automation, README restructuring & clarifications (with dependency table).  
- **2025‑08‑22** — Python audio compare: Torch overlay/diff with start/limit windowing.  
- **2025‑08‑21** — Python video: MiDaS depth‑estimation script.  
- **2025‑08‑20** — C++ audio: WAV → CSV converter.  
- **2025‑08‑18** — Initial setup.

## License
MIT License
