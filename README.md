# Audio-Image-Video-Multimodal AI Toolkit
**Author:** Julia Wen (wendigilane@gmail.com)  
**License:** MIT

This repository contains AI/ML pipelines, tools, and demos for **Audio**, **Image**, **Video**, and **cross-modal multimodal AI applications**. It includes **C++ and Python source code**, **TypeScript and JavaScript React demos** (under `Image/typescript/web` and `Image/javascript/web`), **Streamlit apps**, **database integration**, and **tests** (GoogleTest / Pytest). Each domain provides end-to-end workflows, training/prediction scripts, and interactive demos for research and experimentation. 

## Project Overview
- **Audio:** WAV → CSV, FFT spectra (multiple window types), MLP/NN/RNN models, optional LLM explanations, **live audio denoise with PortAudio and rnnoise**, **2D Conv GAN denoising**.
- **Image:** Crescent detection (HOG+SVM, CNN, Vision Transformer), PyTorch/TensorFlow implementations, dataset generation, **interactive Streamlit apps**, **TypeScript and JavaScript React demos** under `Image/typescript/web` and `Image/javascript/web`.
- **Video:** Motion estimation, frame prediction, stereo disparity, optical flow, trajectory analysis, MiDaS-based depth estimation.
- **Multimodal / Cross-Modal AI:** Graph-based reasoning, LLM explanations, interactive demos, Bayesian apps, and database support.

## Repository Structure
```
Audio/
  cpp/
    src/
    inc/
    tests/
    CMakeLists.txt
  python/
    src/
    tests/
      ai_tools/
    bayesian/
Image/
  python/
    src/
  typescript/
    crescentDemo.ts
    web/
  javascript/
    crescentDemo.js
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
  bayesian/
  graphs/
  llm/
  db/
python_matlab_analogy/
```

## Audio Processing
### C++ Components
- `fft_utils.cpp` — FFT window functions, buffer prep, magnitude and dB helpers.
- `live_audio_denoise.cpp` — Live audio denoising pipeline with PortAudio and rnnoise.
- `wav_freq_csv.cpp` — WAV → CSV and FFT Spectrum CSV, supports multiple FFT windows
- `wav_freq_csv_channelized.cpp` — Multi-channel WAV → CSV + FFT spectrum.
- `wav_to_csv.cpp` — WAV → CSV converter (16-bit, 24-bit, Float32)
- `wav_utils.cpp` — WAV header parsing, sample conversion, endian helpers.
- `wav_writer.cpp` — WAV writing routines for PCM16/24/Float32 with safe I/O.
- Audio/cpp/tests: GoogleTest files for the source above

### Python Components
- CSV comparison: `compare_csv.py`, `comparetorch_csv.py`
- Audio plotting: `comp_plot_wav_diff.py`
- Denoising with GAN: `denoise_gan.py` (YAML config supported)
- Automated FFT workflows: `ai_fft_windowing.py`, `ai_fft_workflow.py`, `ai_test_all_windows.py`
- Neural networks: `nn_module.py` (MLP, NN, RNN)
- Synthetic WAV generation: `generate_wav.py`
- Pytest with AI Tools
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
- TypeScript React demo: `Image/typescript/crescentDemo.ts` using `Image/typescript/web/App.tsx`
- JavaScript React demo: `Image/javascript/crescentDemo.js` using `Image/javascript/web/App.jsx`

## Video Processing
- C++ modules (Video/cpp/VideoEncodingAndVision) for motion, frame prediction, stereo disparity, optical flow, trajectory analysis
video_common/
    inc/
        video_common.h
    src/
        video_common.cpp
video_encoding/
    video_motion_estimation.cpp
    video_frame_prediction.cpp
    video_block_matching.cpp
    video_residual.cpp
video_motion_and_depth/
    video_vio_demo.cpp
    video_trajectory_from_motion.cpp
    video_depth_from_stereo.cpp
- Python moduel (Video/python) depth estimation: `video_depth_midas.py` (MiDaS)
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
- **Bayesian apps:** `Multimodal/bayesian` (`app_bayesian_astar.py`, `app_bayesian_astar_db.py`, `app_bayesian_fft_streamlit_db.py`)
- Graph-based reasoning (A*, knowledge graphs) across embeddings from multiple modalities: `Multimodal/graphs/app_images_astar.py`, `app_images_astar_db.py`
- LLM integration for natural-language explanations: `Multimodal/llm/ai_llm_fft_demo.py`, `Multimodal/llm/test_ai_llm_fft_demo.py`
- Database integration for storing embeddings, predictions, and results
- Streamlit demos for interactive visualization

## python_matlab_analogy
 - Python Matlab analogy examples

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
| Audio C++ | Audio | g++, FFT library, WAV parsing, PortAudio, rnnoise |
| Audio Python | Audio | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile, pandas, openpyxl, python-dotenv |
| Image Python | Image | OpenCV, NumPy, scikit-learn, Matplotlib, joblib, tf.keras, PyTorch |
| Image TypeScript | Image | Node.js, TypeScript, React, Vite |
| Image JavaScript | Image | Node.js, React, Vite |
| Video C++ | Video | g++, OpenCV |
| Video Python | Video | OpenCV, PyTorch, timm, NumPy, Matplotlib |
| Multimodal / AI | Cross-Modal | NetworkX, PyTorch, Streamlit, OpenAI Python SDK, SQLite / other DB |
| Python Matlab Analog | Streamlit, NumPy, Matplotlib, Dash, librosa |

## Notes
- Some parts were developed with AI assistance; core algorithms implemented and verified manually.
- FFT AI demo is offline-first; LLM explanation is optional.
- Multimodal folder connects multiple domains, supports database integration.
- Streamlit apps provide GUI for training, prediction, and output saving.
- TypeScript and JavaScript React demos provide web-based interactive visualization.
- Bayesian tools are modular and integrate with a SQLite database for storing FFT results.
- Live audio denoising requires correct setup of PortAudio and rnnoise libraries.

## Commit / Changelog Highlights
- 2025-12-07 — Audio: live denoise supports different modes
- 2025-12-04 — Audio: overlapping FFT support 
- 2025-12-03 — Added/Updated audio google test files 
- 2025-12-02 — Added bypass to live_audio_denoise
- 2025-12-01 — Updated video processing related c++ files
- 2025-11-25 — Audio live denoise with SPSC and denormal control
- 2025-11-24 — Audio add live denoise with PortAudio and rnnoise
- 2025-11-23 — Audio refactor with common utils
- 2025-11-20 — Added CMakeList for Audio
- 2025-11-18 — Image/javascript javascript React demos for crescent detection
- 2025-11-13 — Added add channelized wav freq to csv
- 2025-11-11 — Python and Matlab analogy examples
- 2025-11-07 — Multimodal db related
- 2025-11-03 — Multimodal/bayesian app_bayesian_astar.py app_bayesian_astar_db.py
- 2025-10-31 — Multimodal/bayesian Bayesian Monte Carlo FFT analysis streamlit app with multimodal.db (sqlite)
- 2025-10-15 — Added Image/typescript typescript React demos for crescent detection
- 2025-09-23 — Added Audio/python/src/denoise_gan.py — Convolutional 2D GAN for audio denoising (YAML config)
- 2025-09-21 — Added Image/python/src/app_predict_crescent_tf.py — Streamlit app for interactive GUI
- 2025-09-20 — Updated Image/python/src/predict_crescent_tf.py — now supports YAML
- 2025-09-17 — Added Multimodal/graphs and Multimodal/llm demos
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

