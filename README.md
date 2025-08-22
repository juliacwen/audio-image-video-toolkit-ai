# Audio & Video Tools

Utilities for working with audio and video data: converting WAV to CSV, comparing waveforms, and running MiDaS depth on videos.

## Features
- **C++**: WAV → CSV (supports PCM **16‑bit**, **24‑bit**, and **IEEE Float32**; outputs `Index,Sample`).
- **Python**: Plot & compare two CSV waveforms; configurable `--start` / `--limit` window.
- **Python (video)**: MiDaS depth‑estimation to MP4 with selectable models.
- **Tests**: `pytest` for Python; **gtest** for C++ with 16/24/32‑bit coverage.

## Project Layout
```
cpp/
  audio/
    wav_to_csv.cpp
  tests/
    test_wav_to_csv.cpp

python/
  audio/
    compare_csv.py
    comparetorch_csv.py
  video/
    video_depth_midas.py
  tests/
    test_compare_csv.py
```

## Setup (Python)
Create and activate a virtual environment, then install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1) WAV → CSV (C++)
Build and run the converter:
```bash
g++ -std=c++17 -O2 -o wav_to_csv cpp/audio/wav_to_csv.cpp
./wav_to_csv input.wav output.csv
```
Output CSV format:
```
Index,Sample
0,0.0
1,-0.001221
...
```

### 2) Compare CSV waveforms (Python)
Both scripts support a windowed comparison via `--start` and `--limit`:

- `--start` → starting **sample index** (0‑based)
- `--limit` → **number of samples** to plot/compare

Visualization:
- **Top plot**: overlay of the two waveforms
- **Bottom plot**: **difference** between them (in **red**)

Run either NumPy/Pandas version or the Torch version:
```bash
python3 python/audio/compare_csv.py samples.csv samples_5.csv --start 1000 --limit 2000
python3 python/audio/comparetorch_csv.py samples.csv samples_5.csv --start 1000 --limit 2000
```

### 3) Video depth (MiDaS)
Depth estimation on a video with model selection:
- `--model` can be **DPT_Large**, **DPT_Hybrid**, or **MiDaS_small**
- `--debug` prints extra info

Example:
```bash
python3 python/video/video_depth_midas.py sample.mp4 out_depth.mp4 --model DPT_Hybrid --debug
```

## Tests

### Python (pytest)
```bash
pytest python/tests
```

### C++ (GoogleTest)
Ensure GoogleTest is installed (e.g., macOS Homebrew):
```bash
brew install googletest
```
Compile and run:
```bash
g++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib     cpp/tests/test_wav_to_csv.cpp -lgtest -lgtest_main -pthread     -o test_wav_to_csv

./test_wav_to_csv
```
The C++ tests generate small **16‑bit**, **24‑bit**, and **float32** WAV files, call `wav_to_csv`, and verify:
- Zero input samples → zero CSV values
- Non‑zero input samples → non‑zero CSV values

## Notes
- All examples use **python3** explicitly.
- CSV readers expect two columns: **Index**, **Sample**.

## Changelog
- **2025‑08‑22** — Python audio tools: compare & visualization improvements.
- **2025‑08‑20** — C++ audio tool: WAV → CSV converter.
- **2025‑08‑18** — Initial utilities overview.

## Author
**Julia Wen**

## License
MIT License
