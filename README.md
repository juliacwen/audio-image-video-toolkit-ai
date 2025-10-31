# Audio-Image-Video-Multimodal AI Toolkit
**Author:** Julia Wen (wendigilane@gmail.com)  
**License:** MIT

This repository contains AI/ML pipelines, tools, and demos for **Audio**, **Image**, **Video**, and **cross-modal multimodal AI applications**. It includes **C++ and Python source code**, **Python Bayesian utilities**, **TypeScript React demos** (under `Image/typescript/web`), **Streamlit apps**, **database integration**, and **tests** (Pytest / GoogleTest). Each domain provides end-to-end workflows, training/prediction scripts, and interactive demos for research and experimentation.

## Project Overview
- **Audio:** WAV → CSV, FFT spectra (multiple window types), MLP/NN/RNN models, optional LLM explanations, **2D Conv GAN denoising**, Bayesian utilities under `Audio/python/bayesian`.
- **Image:** Crescent detection (HOG+SVM, CNN, Vision Transformer), PyTorch/TensorFlow implementations, **dataset generation**, **interactive Streamlit apps**, TypeScript React demos under `Image/typescript/web`.
- **Video:** Motion estimation, frame prediction, stereo disparity, optical flow, trajectory analysis, MiDaS-based depth estimation.
- **Multimodal / Cross-Modal AI:** Graph-based reasoning, LLM explanations, interactive demos, and database support for embeddings/outputs.

## Key Features

### Audio
- WAV → CSV conversion, FFT spectra with multiple window types (rectangular, Hann, Hamming, Blackman)
- AI-assisted FFT workflows: MLP, NN, RNN
- Optional LLM explanations for FFT results
- Convolutional 2D GAN for audio denoising
- Bayesian utilities for probabilistic signal analysis (`Audio/python/bayesian`)

### Image
- Crescent detection using classical (HOG + SVM) and deep learning approaches
- CNN, Vision Transformer, TensorFlow & PyTorch implementations
- Dataset generation tools
- Updated TensorFlow CNN (`predict_crescent_tf.py`) supports YAML config, outputs `_tf.png` labeled images, and saves results to `test_output/predict_crescent_tf_result.txt`
- Streamlit app (`app_predict_crescent_tf.py`) for interactive GUI training, image upload, predictions, and output saving
- TypeScript React demo: `Image/typescript/crescentDemo.ts` and web folder `Image/typescript/web`

### Video
- C++ modules: motion estimation, frame prediction, stereo disparity, optical flow, trajectory analysis
- Python depth estimation using MiDaS

### Multimodal / Cross-Modal AI
- Graph-based reasoning (A*, knowledge graphs) across embeddings from multiple modalities
- LLM integration for natural-language explanations
- Database integration for storing embeddings, predictions, and results
- Streamlit demos for interactive visualization

## Repository Structure
```
Audio/
  cpp/
    src/
    tests/
  python/
    src/
    bayesian/
    tests/
      ai_tools/
Image/
  python/
    src/
      dataset/
  typescript/
    crescentDemo.ts
    web/
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
  db/           # Database integration
```

## Audio Processing
### C++ Components
- `Audio/cpp/src/wav_to_csv.cpp` — WAV → CSV (16-bit, 24-bit, Float32)
- `Audio/cpp/src/wav_freq_csv.cpp` — WAV → CSV and FFT Spectrum CSV, supports multiple FFT windows
- Tests: GoogleTest `test_wav_to_csv.cpp`, `test_wav_freq_csv.cpp`

### Python Components
- CSV comparison: `compare_csv.py`, `comparetorch_csv.py`
- Audio plotting: `comp_plot_wav_diff.py`
- Denoising with GAN: `denoise_gan.py` (YAML config supported)
- Pytest with AI Tools
- Automated FFT workflows: `ai_fft_windowing.py`, `ai_fft_workflow.py`, `ai_test_all_windows.py`
- Neural networks: `nn_module.py` (MLP, NN, RNN)
- Synthetic WAV generation: `generate_wav.py`
- Bayesian utilities: `Audio/python/bayesian`
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
- TypeScript React demo: `Image/typescript/crescentDemo.ts` using `Image/typescript/web`

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
- Database support for embeddings, predictions, results
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
| Audio Bayesian | Audio | NumPy, SciPy, PyTorch, Pandas |
| Audio AI Tools | Audio/AI | NumPy, Pandas, PyTorch, openpyxl, python-dotenv, (optional: LangChain, OpenAI Python SDK) |
| Image Python | Image | OpenCV, NumPy, scikit-learn, Matplotlib, joblib, tf.keras, PyTorch |
| Image TypeScript | Image | Node.js, TypeScript, React, Vite |
| Video C++ | Video | g++, OpenCV |
| Video Python | Video | OpenCV, PyTorch, timm, NumPy, Matplotlib |
| Multimodal / AI | Cross-Modal | NetworkX, PyTorch, Streamlit, OpenAI Python SDK, SQLite / other DB |

## Notes
- Some parts were developed with AI assistance; core algorithms implemented and verified manually.
- FFT AI demo is offline-first; LLM explanation is optional.
- Multimodal folder connects multiple domains, supports database integration.
- Streamlit apps provide GUI for training, prediction, and output saving.
- TypeScript React demos provide web-based interactive visualization.

## Commit / Changelog Highlights
- 2025-10-31 — Audio/python/bayesian Bayesian Monte Carlo FFT analysis streamlit app with multimodal.db (sqlite)
- 2025-10-15 — Added Image/typescript TypeScript React demo for crescent detection and database support
- 2025-09-23 — Added Audio/python/src/denoise_gan.py — Convolutional 2D GAN for audio denoising (YAML config)
- 2025-09-21 — Added Image/python/src/app_predict_crescent_tf.py — Streamlit app for interactive GUI
- 2025-09-20 — Updated Image/python/src/predict_crescent_tf.py — now supports YAML
- 2025-09-17 — Added Multimodal/graphs/app_images_astar.py, Multimodal/llm/ai_llm_fft_demo.py, Multimodal/llm/test_ai_llm_fft_demo.py
- 2025-09-14 — Update Audio pytest to save all output (including PNG) to test_output
- 2025-09-13 — Update Audio pytest coverage for MLP, NN, RNN
- 2025-09-11 — Added PyTorch NN model to AI FFT workflow and updated tests
- 2025-09-10 — Added AI FFT workflow with LLM demo
- 2025-09-08 — Added Image PyTorch CNN model
- 2025-09-07 — Modernize Audio C++
- 2025-09-05 — Refactor Video C++
- 2025-09-04 — Added Video C++ modules for motion, encoding, depth estimation; updated structure
- 2025-08-29 — Added crescent detection scripts; repo restructure
- 2025-08-27 — Added AI FFT prediction workflow, Python/pytest validation
- 2025-08-25 — Updated wav_freq_csv with FFT window support; AI-assisted tests
- 2025-08-23 — Added WAV→CSV, spectrum comparison, video pytest
- 2025-08-22 — Python audio compare: Torch overlay/diff with start/limit windowing
- 2025-08-21 — Python video: MiDaS depth-estimation script
- 2025-08-20 — C++ audio: WAV → CSV converter
- 2025-08-18 — Initial setup

## License
MIT License  
© 2025 Julia Wen (wendigilane@gmail.com)