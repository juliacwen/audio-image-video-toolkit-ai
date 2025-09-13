# Audio-Image-Video-Toolkit-AI/Audio
**Author:** Julia Wen (<wendigilane@gmail.com>)

## Project Overview
This repository section contains projects for **Audio** processing, including C++ and Python tools, automated tests, and AI-assisted workflows.

## Features

- Converting WAV to CSV, generating FFT spectra with multiple window types.
- Running end-to-end AI-assisted FFT workflows that generate test WAVs, train MLP/RNN/PyTorch NN models, and predict tone probabilities.
- Performing automated FFT windowing tests with rectangular, Hann, Hamming, and Blackman windows.
- Optional **LLM explanations** using OpenAI if API key is set.

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
```

## Audio Processing

### C++ Components
- `Audio/cpp/src/wav_to_csv.cpp`: WAV → CSV (supports PCM 16-bit, 24-bit, IEEE Float32; outputs `Index,Sample`).
- `Audio/cpp/src/wav_freq_csv.cpp`: WAV → CSV **and** FFT Spectrum CSV (`Index,Sample` and `Frequency,Magnitude`). Supports selectable FFT windows: rectangular, Hann, Hamming, Blackman.

#### Tests
- `Audio/cpp/tests/test_wav_to_csv.cpp` — GoogleTest
- `Audio/cpp/tests/test_wav_freq_csv.cpp` — GoogleTest

### Python Components
- `Audio/python/src/compare_csv.py` — compare two time-domain WAV CSVs. Uses NumPy, Pandas, Matplotlib.
- `Audio/python/src/comparetorch_csv.py` — compare time-domain or spectrum CSVs using PyTorch tensors.
- `Audio/python/src/comp_plot_wav_diff.py` — compare two WAV audio files using torchaudio and Matplotlib.

#### Pytest and AI-assisted Tools
- `Audio/python/tests/test_ai_fft_windowing.py` — automated FFT windowing tests.
- `Audio/python/tests/test_ai_fft_workflow.py` — end-to-end AI FFT workflow test.
- `Audio/python/tests/ai_tools/ai_fft_windowing.py` — apply different FFT windows to audio segments.
- `Audio/python/tests/ai_tools/ai_fft_workflow.py` — AI-assisted workflow: AI FFT workflow using nn_module (MLP + RNN + NN)
- `Audio/python/tests/ai_tools/ai_test_all_windows.py` — automated testing for all FFT window types.
- `Audio/python/tests/ai_tools/generate_wav.py` — generate synthetic WAV files.
- `Audio/python/tests/ai_tools/nn_module.py` — MLP, NN, RNN for tone prediction.

### End-to-end AI FFT Test Workflow
- `Audio/python/tests/ai_llm_fft_demo.py` — AI FFT demo with optional LLM explanation.
- `Audio/python/tests/test_comp_plot_wav_diff.py` — pytest for comp_plot_wav_diff.py
- `Audio/python/tests/test_compare_csv.py` — pytest for compare_csv.py
- `Audio/python/tests/test_comparetorch.py` — pytest for comparetorch_csv.py

**Workflow Details:**
1. Generate WAVs for each bit depth (16bit, 24bit, float32) and create CSV + FFT spectrum.
2. Train models on synthetic spectra:
   - MLP using `train_MLP` from nn_module.py
   - RNN using `train_rnn` from nn_module.py
   - PyTorch NN using `train_nn` from nn_module.py
3. Make predictions for each spectrum with all models (MLP, RNN, PyTorch NN).
4. Validate outputs: probability arrays per model match number of classes, sum to ~1.
5. Save predictions to `test_output/predictions.txt` with format `filename,model,probs`.

**File locations for workflow:**
- WAVs: `Audio/cpp/tests/ai_tools/test_output/tone_<suffix>.wav`
- CSVs: `Audio/cpp/tests/ai_tools/test_output/tone_<suffix>.csv`
- FFT Spectra: `Audio/cpp/tests/ai_tools/test_output/tone_<suffix>_spectrum.csv`
- Predictions: `Audio/cpp/tests/ai_tools/test_output/predictions.txt`

## Optional LLM Explanation

- Example script: `ai_llm_fft_demo.py`
- Requires OpenAI API key (`OPENAI_API_KEY`) to generate plain-language explanations of FFT results.
- Offline FFT analysis and plotting function normally if no API key is set.

## Dependencies

| Script / Tool   | Libraries / Tools |
|-----------------|-----------------|
| Audio C++       | g++, standard C++ libraries, FFT, WAV parsing |
| Audio Python    | NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, SoundFile, pandas, openpyxl, python-dotenv |
| Audio AI Tools  | NumPy, Pandas, PyTorch, openpyxl, python-dotenv, (optional: LangChain, OpenAI Python SDK) |

## Build & Run

**C++ compilation**
```bash
make -C Audio/cpp
```
**Python scripts**
```bash
cd Audio/python/src
python compare_csv.py
python ai_fft_demo.py ../../test_files/readme_sample.wav
```

## Notes
- Some source code assisted with AI tools, but core logic implemented and verified manually.
- The FFT AI demo is offline-first; LLM explanation is optional.

## License
MIT License

