# Audio-Image-Video-Toolkit-AI/Audio
**Author:** Julia Wen (<wendigilane@gmail.com>)

## Project Overview
This repository section contains projects for **Audio** processing, including C++ and Python tools, automated tests, AI-assisted workflows, and Bayesian inference modules.

## Features

- Convert WAV files to CSV, generate FFT spectra with multiple window types.
- End-to-end AI-assisted FFT workflows: generate synthetic WAVs, train MLP/RNN/PyTorch NN models, and predict tone probabilities.
- Perform automated FFT windowing tests with rectangular, Hann, Hamming, and Blackman windows.
- Train and infer **GAN-based audio denoising** models using YAML configuration.
- Optional **LLM explanations** for FFT results (requires OpenAI API key).
- Modular **Bayesian** utilities for probabilistic signal analysis, including a SQLite database for storing FFT results.

## Repository Structure

```text
Audio/
  cpp/
    src/
    tests/
  python/
    src/
    tests/
      ai_tools/
    bayesian/
```

---

## Audio Processing

### C++ Components
- **`Audio/cpp/src/wav_to_csv.cpp`** — Converts WAV → CSV (supports PCM 16-bit, 24-bit, IEEE Float32; outputs `Index,Sample`).
- **`Audio/cpp/src/wav_freq_csv.cpp`** — Converts WAV → CSV **and** FFT Spectrum CSV (`Index,Sample` and `Frequency,Magnitude`).  
  Supports FFT windows: rectangular, Hann, Hamming, Blackman.

#### Tests
- **`Audio/cpp/tests/test_wav_to_csv.cpp`** — GoogleTest for `wav_to_csv`.
- **`Audio/cpp/tests/test_wav_freq_csv.cpp`** — GoogleTest for `wav_freq_csv`.

---

### Python Components

#### Core Scripts (`Audio/python/src/`)
- **`compare_csv.py`** — Compare two time-domain WAV CSVs using NumPy, Pandas, and Matplotlib.
- **`comparetorch_csv.py`** — Compare time-domain or spectrum CSVs using PyTorch tensors.
- **`comp_plot_wav_diff.py`** — Compare two WAV audio files directly with torchaudio and Matplotlib.
- **`denoise_gan.py`** — **Conv2D U-Net GAN for audio denoising.**  
  Uses a YAML configuration file (`config_denoise_gan.yaml`) to specify dataset paths, model hyperparameters, and training options.

---

#### AI-assisted FFT Tools (`Audio/python/tests/ai_tools/`)
- **`ai_fft_windowing.py`** — Apply different FFT windows to audio segments.
- **`ai_fft_workflow.py`** — End-to-end AI-assisted FFT workflow using MLP, RNN, and NN modules.
- **`ai_test_all_windows.py`** — Automated testing across all FFT window types.
- **`generate_wav.py`** — Generate synthetic WAVs for testing.
- **`nn_module.py`** — Neural network definitions: MLP, NN, and RNN for tone prediction.

#### Pytest Suites (`Audio/python/tests/`)
- **`test_ai_fft_windowing.py`** — Automated FFT windowing tests.
- **`test_ai_fft_workflow.py`** — End-to-end AI FFT workflow tests.
- **`test_comp_plot_wav_diff.py`** — pytest for `comp_plot_wav_diff.py`.
- **`test_compare_csv.py`** — pytest for `compare_csv.py`.
- **`test_comparetorch.py`** — pytest for `comparetorch_csv.py`.
- **`ai_llm_fft_demo.py`** — FFT analysis demo with optional LLM explanation (requires OpenAI API key).

---
##### Denoise GAN Usage:
```bash
# Check dataset
python3 denoise_gan.py --config config_denoise_gan.yaml --check_dataset

# Train model
python3 denoise_gan.py --config config_denoise_gan.yaml --train

# Inference (denoise WAV)
python3 denoise_gan.py --config config_denoise_gan.yaml --infer     --noisy_wav data/noisy/sample_noisy.wav     --out_wav data/cleaned/sample_denoised.wav
```
---

### Bayesian Tools (`Audio/python/bayesian/`)
This module group provides Bayesian inference utilities for probabilistic audio analysis, uncertainty modeling, and spectral prior estimation.

- Uses **SQLite database** `multimodal.db` at the project root to store FFT results and Monte Carlo samples.
- Capabilities include:
  - Posterior estimation for FFT magnitude distributions.
  - Bayesian smoothing of noisy spectra.
  - Probabilistic tone classification from FFT magnitude vectors.

**Streamlit Demo Usage:**
```bash
streamlit run Audio/python/bayesian/bayesian_fft_app.py
```

---

## End-to-End AI FFT Test Workflow
1. Generate WAVs for each bit depth (16-bit, 24-bit, float32) and create CSV + FFT spectrum.
2. Train AI models on synthetic spectra:
   - MLP → `train_MLP` from `nn_module.py`
   - RNN → `train_rnn` from `nn_module.py`
   - PyTorch NN → `train_nn` from `nn_module.py`
3. Make predictions for each spectrum with all models (MLP, RNN, NN).
4. Validate outputs — probability arrays per model match number of classes and sum to ≈1.
5. Save predictions to `Audio/test_output/predictions.txt`.

**File locations for workflow:**
- WAVs: `Audio/test_output/tone_<suffix>.wav`
- CSVs: `Audio/test_output/tone_<suffix>.csv`
- FFT Spectra: `Audio/test_output/<filename>_ai_spectrum.csv` and `<filename>_ai.png`
- Predictions: `Audio/test_output/predictions.txt`

---

## Optional LLM Explanation

- Example script: `ai_llm_fft_demo.py`
- Requires OpenAI API key (`OPENAI_API_KEY`) to generate natural-language explanations of FFT results.
- FFT analysis and plotting still function offline when no key is set.

---

## Dependencies

| Component | Libraries / Tools |
|------------|------------------|
| **Audio C++** | g++, FFT, WAV parsing |
| **Audio Python** | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile, pandas, openpyxl, python-dotenv |
| **AI Tools** | NumPy, Pandas, PyTorch, openpyxl, python-dotenv, (optional: LangChain, OpenAI SDK) |
| **Denoise GAN** | PyTorch, NumPy, PyDub, PyYAML |
| **Bayesian Tools** | NumPy, SciPy, PyMC, matplotlib (planned), SQLite database for FFT results |

---

## Build & Run

**C++ Compilation**
```bash
make -C Audio/cpp
```

**Python Scripts**
```bash
cd Audio/python/src
python compare_csv.py
python denoise_gan.py --config config_denoise_gan.yaml --train
```

**Run Full AI FFT Workflow**
```bash
pytest Audio/python/tests/test_ai_fft_workflow.py -v
```

**Streamlit Bayesian Demo**
```bash
streamlit run Audio/python/bayesian/bayesian_fft_app.py
```

---

## Notes
- Core logic implemented and verified manually; AI tools assisted in code generation.
- The FFT AI demo and GAN denoiser are **offline-first**; LLM and YAML options are **optional**.
- Bayesian tools are modular and integrate with a SQLite database for storing FFT results.

---

## License
MIT License
