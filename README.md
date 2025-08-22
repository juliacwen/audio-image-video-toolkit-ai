# Audio Video Tools

This project provides utilities for working with audio and video files, including
conversion of WAV files to CSV, waveform comparison, and depth estimation from video.

## Requirements

Install Python dependencies into a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For C++ components you need a modern compiler (C++17) and [GoogleTest](https://github.com/google/googletest).  
On macOS with Homebrew:

```bash
brew install googletest
```

## Usage

### WAV to CSV (C++)

Convert a WAV file to CSV (supports PCM 16‑bit, PCM 24‑bit, and IEEE Float32 WAVs):

```bash
g++ -std=c++17 -o wav_to_csv cpp/audio/wav_to_csv.cpp
./wav_to_csv input.wav output.csv
```

The output CSV contains two columns: `Index` and `Sample`.

### Compare CSV (Python)

Compare two CSV files and visualize the difference:

```bash
python3 python/audio/compare_csv.py samples.csv samples_5.csv --start 100 --limit 1000
```

Overlay plot (top): both signals.  
Difference plot (bottom): residuals highlighted in red.

You can also use the Torch implementation:

```bash
python3 python/audio/comparetorch_csv.py samples.csv samples_5.csv --start 100 --limit 1000
```

### Video Depth Estimation (Python)

Run MiDaS depth estimation on a video:

```bash
python3 python/video/video_depth_midas.py sample.mp4 out_depth.mp4 --debug
```

## Running Tests

### Python Tests

Python tests use **pytest**. Make sure your virtual environment is activated and dependencies installed:

```bash
pip install -r requirements.txt
pytest python/tests
```

This runs the CSV comparison tests and validates that sample‑to‑CSV conversion works as expected.

### C++ Tests

C++ tests use **GoogleTest (gtest)**. They generate temporary WAV files in 16‑bit PCM, 24‑bit PCM, and IEEE float32, then run the `wav_to_csv` converter and check that:

- Zero input samples produce zero CSV values  
- Non‑zero input samples produce non‑zero CSV values  

#### Build the Test

```bash
g++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib     cpp/tests/test_wav_to_csv.cpp -lgtest -lgtest_main -pthread     -o test_wav_to_csv
```

#### Run the Test

```bash
./test_wav_to_csv
```

Expected output:

```
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from WavToCsvTest
[ RUN      ] WavToCsvTest.Converts16BitWav
[       OK ] WavToCsvTest.Converts16BitWav (5 ms)
[ RUN      ] WavToCsvTest.Converts24BitWav
[       OK ] WavToCsvTest.Converts24BitWav (6 ms)
[ RUN      ] WavToCsvTest.ConvertsFloat32Wav
[       OK ] WavToCsvTest.ConvertsFloat32Wav (5 ms)
[----------] 3 tests from WavToCsvTest (16 ms total)

[  PASSED  ] 3 tests.
```

## License

MIT License
