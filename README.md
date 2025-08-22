# Audio & Video Tools

This project provides a collection of small utilities for working with **audio** and **video** data.
08-18-25
- **C++ audio tool**: Convert WAV files to CSV sample data.
08-20-25
- **Python audio tools**: Compare and visualize CSV audio data.
08-22-25
- **Python video tool**: Generate depth-estimation videos using MiDaS models.
---
## 🚀 Setup
### 1. Clone and enter project
```bash
git clone <your-repo-url>
cd audio_video_tools_correct
```
### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
---
## 🎵 C++ Audio Tool
Convert `.wav` files to `.csv` sample data.
### Build
```bash
cd cpp/audio
# build the converter
g++ -std=c++17 -O2 -Wall -Wextra -o wav_to_csv wav_to_csv.cpp

```
### Run
```bash
./wav_to_csv input.wav output.csv
```
---
## 🎶 Python Audio Tools
Compare CSV audio sample files and visualize differences.

### Compare with matplotlib
```bash
cd python/audio
python3 compare_csv.py samples.csv samples_5.csv --start 0 --limit 100

--start → starting sample index
--limit → number of samples to plot
- **Top plot**: overlay of the two waveforms
- **Bottom plot**: difference between them (in red)
```
### Compare with PyTorch backend
```bash
python3 comparetorch_csv.py samples.csv samples_5.csv --start 0 --limit 100
```
This version uses PyTorch tensors for faster handling of large CSVs.
---
## 🎥 Video Depth Tool
Generate a depth-estimation video from an input video using **MiDaS**.

### Run
```bash
cd python/video
python3 video_depth_midas.py sample.mp4 out_depth.mp4 --model DPT_Large --debug
```
- `--model` can be `DPT_Large`, `DPT_Hybrid`, or `MiDaS_small`
- `--debug` prints extra information while processing

### Example Output
- Takes `sample.mp4`
- Produces `out_depth.mp4` with depth maps colorized using `COLORMAP_MAGMA`
---

## 📦 Requirements
See [`requirements.txt`](requirements.txt). Key libraries:
- PyTorch (torch, torchvision, torchaudio)
- OpenCV
- matplotlib, numpy, scipy
- timm (for MiDaS models)
- imageio + imageio-ffmpeg
---

## 📂 Project Structure
```
audio_video_tools_correct/
├── requirements.txt
├── cpp/
│   └── audio/
│       ├── wav_to_csv.cpp
│       ├── wav_to_csv_fix.cpp
│       └── ...
├── python/
│   ├── audio/
│   │   ├── compare_csv.py
│   │   ├── comp_plot_wav_diff.py
│   │   └── comparetorch_csv.py
│   └── video/
│       └── video_depth_midas.py
```

---
## ✨ Notes
- Use `.avi`if MP4 codecs fail with OpenCV. With `imageio-ffmpeg`, `.mp4` writing is always supported.
- For consistent environments, recreate the venv and run `pip install -r requirements.txt`.
---
## Author
Julia Wen jwenhub@gmail.com

## 📜 License
MIT License (adjust as needed).
