# Audio-Image-Video-Toolkit-AI/Audio
**Author:** Julia Wen (<wendigilane@gmail.com>)

## Project Overview
This repository section contains projects for **Audio** processing, including C++ and Python tools, automated tests, AI-assisted workflows, and Bayesian inference modules.

## Features
- C++ components include WAV reading/writing, FFT computation, and live audio denoising.
- Convert WAV files to CSV, generate FFT spectra with multiple window types.
- Live audio denoise with PortAudio and rnnoise, bypass option, SPSC, denormal control
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
    inc/
    tests/
    CMakeLists.txt
  python/
    src/
    tests/
      ai_tools/
    bayesian/
```

---

## Audio Processing

### C++ Components
- **Headers (`Audio/cpp/inc/`)**
  - `SPSCFloatBuffer.h` — Single-producer/single-consumer lock-free float buffer.
  - `denormal_control.h` — Prevent CPU denormals and small-noise injection helpers.
  - `error_codes.h` — Error code enumerations and helper functions.
  - `fft_utils.h` — FFT window functions, buffer prep, magnitude and dB helpers.
  - `wav_utils.h` — WAV header parsing, sample conversion, endian helpers.
  - `wav_writer.h` — WAV writing routines for PCM16/24/Float32 with safe I/O.

- **Sources (`Audio/cpp/src/`)**
  - `fft_utils.cpp` — Implements window functions and FFT helpers.
  - `live_audio_denoise.cpp` — Live audio denoising pipeline.
  - `wav_comp_diff.cpp` — Compare WAVs or WAV CSVs and output differences.
  - `wav_freq_csv.cpp` — WAV → CSV + FFT spectrum.
  - `wav_freq_csv_channelized.cpp` — Multi-channel WAV → CSV + FFT spectrum.
  - `wav_to_csv.cpp` — WAV → CSV converter.
  - `wav_utils.cpp` — Implements WAV reading and sample conversion.
  - `wav_writer.cpp` — Implements WAV writing routines.

- **Tests (`Audio/cpp/tests/`)**
  - `test_live_audio_denoise.cpp` — Verifies live denoising.
  - `test_wav_freq_csv.cpp` — Tests `wav_freq_csv` spectra correctness.
  - `test_wav_freq_csv_multi_channel.cpp` — Multi-channel spectrum tests.
  - `test_wav_to_csv.cpp` — WAV → CSV conversion tests.
  - `test_wav_utils.cpp` — WAV parsing and edge case tests.
  - `test_wav_writer.cpp` — Round-trip WAV writing tests.
  - `test_wav_writer_multichannel.cpp` — Multi-channel WAV writing tests.

#### Build Instructions
Build C++ components using CMake:
```bash
cd Audio/cpp
mkdir -p build && cd build
cmake ..
make
```
This will produce the test executables and any library targets.

### Python Components

#### Core Scripts (`Audio/python/src/`)
- `compare_csv.py` — Compare two time-domain WAV CSVs using NumPy, Pandas, Matplotlib.
- `comparetorch_csv.py` — Compare time-domain or spectrum CSVs using PyTorch.
- `comp_plot_wav_diff.py` — Compare two WAV audio files directly.
- `denoise_gan.py` — Conv2D U-Net GAN for audio denoising (config via YAML). Requires **PortAudio** and **rnnoise**.

#### AI-assisted FFT Tools (`Audio/python/tests/ai_tools/`)
- `ai_fft_windowing.py` — Apply different FFT windows.
- `ai_fft_workflow.py` — End-to-end AI-assisted FFT workflow.
- `ai_test_all_windows.py` — Automated tests across all FFT windows.
- `generate_wav.py` — Generate synthetic WAVs for testing.
- `nn_module.py` — Neural network definitions: MLP, NN, RNN.

#### Pytest Suites (`Audio/python/tests/`)
- `test_ai_fft_windowing.py`, `test_ai_fft_workflow.py`, `test_comp_plot_wav_diff.py`, `test_compare_csv.py`, `test_comparetorch.py`, `ai_llm_fft_demo.py`

##### Denoise GAN Usage:
```bash
python3 denoise_gan.py --config config_denoise_gan.yaml --check_dataset
python3 denoise_gan.py --config config_denoise_gan.yaml --train
python3 denoise_gan.py --config config_denoise_gan.yaml --infer --noisy_wav data/noisy/sample_noisy.wav --out_wav data/cleaned/sample_denoised.wav
```

### Bayesian Tools (`Audio/python/bayesian/`)
- Bayesian inference for audio spectra, uncertainty modeling, spectral prior estimation.
- Uses SQLite `multimodal.db` to store FFT results.
- Streamlit demo: `streamlit run Audio/python/bayesian/bayesian_fft_app.py`

## End-to-End AI FFT Test Workflow
1. Generate WAVs for 16-bit, 24-bit, float32 → CSV + FFT.
2. Train AI models (MLP, RNN, PyTorch NN).
3. Predict spectrum classes.
4. Validate outputs (probabilities sum ≈1).
5. Save predictions to `Audio/test_output/predictions.txt`.

## Optional LLM Explanation
- `ai_llm_fft_demo.py` generates natural-language explanations using OpenAI API key.

## Dependencies
| Component | Libraries / Tools |
|------------|------------------|
| Audio C++ | g++, FFT library, WAV parsing, PortAudio, rnnoise |
| Audio Python | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile, pandas, openpyxl, python-dotenv |
| AI Tools | NumPy, Pandas, PyTorch, openpyxl, python-dotenv, optional LangChain/OpenAI SDK |
| Denoise GAN | PyTorch, NumPy, PyDub, PyYAML, PortAudio, rnnoise |
| Bayesian Tools | NumPy, SciPy, PyMC, matplotlib, SQLite |

## Build & Run
**C++:**
```bash
cd Audio/cpp
mkdir -p build && cd build
cmake ..
make
```
**Python:** run scripts in `Audio/python/src/`
**Full AI FFT Workflow:** `pytest Audio/python/tests/test_ai_fft_workflow.py -v`
**Streamlit Demo:** `streamlit run Audio/python/bayesian/bayesian_fft_app.py`

## Notes
- Core logic implemented and verified manually; AI tools assisted in some code generation.
- The FFT AI demo and GAN denoiser are **offline-first**; LLM and YAML options are **optional**.
- Bayesian tools are modular and integrate with a SQLite database for storing FFT results.
- Live audio denoising requires correct setup of PortAudio and rnnoise libraries.

## License
MIT License

