# Audio-Image-Video-Multimodal AI Toolkit
**Author:** Julia Wen (wendigilane@gmail.com)  
**License:** MIT

This repository provides AI/ML pipelines and tools for Audio, Image, Video, and cross-modal multimodal AI applications. Each top-level directory contains source code (C++ and Python) and tests (Pytest / GoogleTest) designed for research, experimentation, and interactive demos.

## Project Overview
The repository contains projects for Audio, Image, Video processing, and cross-modal AI/ML tools. Each domain contains source code (C++ and Python) and tests intended for development and experimentation with AI models.

## Key Features

### Audio
- WAV → CSV conversion, FFT spectra with multiple window types (rectangular, Hann, Hamming, Blackman)
- AI-assisted FFT workflows: MLP, RNN, PyTorch NN
- Optional LLM explanations for FFT results

### Image
- Crescent detection using classical (HOG + SVM) and deep learning approaches
- CNN, Vision Transformer, TensorFlow & PyTorch implementations
- Dataset generation tools
- Updated TensorFlow CNN (`predict_crescent_tf.py`) supports YAML config, outputs `_tf.png` labeled images, and saves results to `test_output/predict_crescent_tf_result.txt`
- Streamlit app (`app_predict_crescent_tf.py`) for live training, upload, predictions, and output saving

### Video
- C++ modules: motion estimation, frame prediction, stereo disparity, optical flow, trajectory analysis
- Python depth estimation using MiDaS

### Multimodal / Cross-Modal AI
- Graph-based reasoning (A*, knowledge graphs) across embeddings from multiple modalities
- LLM integration for natural-language explanations
- Streamlit demos for interactive visualization

## Repository Structure
```
Audio/
  cpp/
    src/
    tests/
  python/
    src/
    tests/
      ai_tools/
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
Multimodal/
  graphs/       # A*, knowledge graph demos
  llm/          # LLM integration scripts
```

## Audio Processing
### C++ Components
- `Audio/cpp/src/wav_to_csv.cpp` — WAV → CSV (16-bit, 24-bit, Float32)
- `Audio/cpp/src/wav_freq_csv.cpp` — WAV → CSV and FFT Spectrum CSV, supports multiple FFT windows
- Tests: GoogleTest `test_wav_to_csv.cpp`, `test_wav_freq_csv.cpp`

### Python Components
- CSV comparison: `compare_csv.py`, `comparetorch_csv.py`
- Audio plotting: `comp_plot_wav_diff.py`
- Pytest with AI Tools
- Automated FFT workflows: `ai_fft_windowing.py`, `ai_fft_workflow.py`, `ai_test_all_windows.py`
- Neural networks: `nn_module.py` (MLP, NN, RNN)
- Synthetic WAV generation: `generate_wav.py`
- End-to-end AI FFT demo: `ai_llm_fft_demo.py` (optional LLM explanation)

## Image Processing
### Crescent detection scripts
- `predict_crescent.py` — SVM + HOG + augmentation
- `predict_crescent_pytorch.py` — PyTorch SVM classifier
- `predict_crescent_pytorch_cnn.py` — CNN classifier
- `predict_crescent_tf.py` — TensorFlow CNN; YAML config support, labeled `_tf.png` outputs, results saved in `test_output/`
- `predict_crescent_tf_classic.py` — TensorFlow wrapper for HOG+SVM
- `detect_crescent_classical.py` — Classical HOG + SVM
- `predict_crescent_vit.py` — Vision Transformer
- Streamlit app: `app_predict_crescent_tf.py` for interactive training/predictions
- Dataset generation: `generate_crescent_images.py`

## Video Processing
- C++ modules for motion, frame prediction, stereo disparity, optical flow, trajectory analysis
- Python depth estimation: `video_depth_midas.py` (MiDaS)
- C++ compilation:
```bash
make -C Video/cpp/VideoEncodingAndVision
```
- Python scripts:
```bash
cd Video/python/src
python video_depth_midas.py --video path/to/video.mp4
```

## Multimodal / Cross-Modal AI
- Graph-based reasoning (A*, knowledge graphs) across embeddings
- LLM integration for explanations
- Streamlit demo:
```bash
cd Multimodal/graphs
streamlit run astar_demo.py
```

## Setup
1. Clone the repository
```bash
git clone https://github.com/juliacwen/audio-image-video-toolkit-ai/
cd audio-image-video-toolkit-ai
```
2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate     # Windows PowerShell
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Dependencies by Module
| Module | Domain | Libraries / Tools |
|--------|--------|-----------------|
| Audio C++ | Audio | g++, FFT, WAV parsing |
| Audio Python | Audio | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile, pandas, openpyxl, python-dotenv |
| Audio AI Tools | Audio/AI | NumPy, Pandas, PyTorch, openpyxl, python-dotenv, (optional: LangChain, OpenAI Python SDK) |
| Image Python | Image | OpenCV, NumPy, scikit-learn, Matplotlib, joblib, tf.keras, PyTorch |
| Video C++ | Video | g++, OpenCV |
| Video Python | Video | OpenCV, PyTorch, timm, NumPy, Matplotlib |
| Multimodal / AI | Cross-Modal | NetworkX, PyTorch, Streamlit, OpenAI Python SDK |

## Notes
- Some code assisted by AI tools; core logic verified manually
- FFT AI demo is offline-first; LLM explanation is optional
- Multimodal folder contains AI/ML tools connecting multiple domains

## Changelog
- 2025‑09‑20 — Updated Image folder: `predict_crescent_tf.py` now supports YAML config, labeled `_tf.png` images, and results saved in `test_output/`; Streamlit app `app_predict_crescent_tf.py` added
- 2025‑09‑17 — Added Multimodal/graphs/app_images_astar.py, Multimodal/llm/ai_llm_fft_demo.py, Multimodal/llm/test_ai_llm_fft_demo.py
- 2025‑09‑14 — Update Audio pytest to save all output (including png) to test_output
- 2025‑09‑13 — Update Audio pytest coverage for MLP, NN, RNN
- 2025‑09‑11 — Added PyTorch NN model to AI FFT workflow and updated tests
- 2025‑09‑10 — Added AI FFT flow with LLM demo
- 2025‑09‑08 — Added image processing with PyTorch CNN model
- 2025‑09‑07 — Modernize Audio C++
- 2025‑09‑05 — Refactor Video C++
- 2025‑09‑04 — Added Video C++ modules for motion, encoding, depth estimation; updated structure
- 2025‑08‑29 — Added crescent detection scripts; repo restructure
- 2025‑08‑27 — Added AI FFT prediction workflow, Python/pytest validation
- 2025‑08‑25 — Updated wav_freq_csv with FFT window support; AI-assisted tests
- 2025‑08‑23 — Added WAV→CSV, spectrum comparison, video pytest
- 2025‑08‑22 — Python audio compare: Torch overlay/diff with start/limit windowing
- 2025‑08‑21 — Python video: MiDaS depth-estimation script
- 2025‑08‑20 — C++ audio: WAV → CSV converter
- 2025‑08‑18 — Initial setup

## License
MIT License  
© 2025 Julia Wen (wendigilane@gmail.com)

