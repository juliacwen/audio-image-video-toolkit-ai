
#!/usr/bin/env python3
"""
ai_llm_fft_demo.py - Windowed FFT demo with LLM integration and MLOps-style logging.

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-15
- Always runs offline FFT (multiple windows), saves Excel (separate sheets), plots average spectra.
- Prints WAV metadata (sampling rate, samples, channels, dtype).
- Reports runtime status for LangChain, MLOps (dotenv), LLMOps (OpenAI key).
- Attempts LLM explanation only if LangChain and OPENAI_API_KEY present.
- Writes workflow JSON log to specified output_dir.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf

# ------------------------------
# Optional components
# ------------------------------
DOTENV_LOADED = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_LOADED = True
except Exception:
    DOTENV_LOADED = False

LANGCHAIN_AVAILABLE = False
ChatOpenAI_impl = None
try:
    from langchain.chat_models import ChatOpenAI as _ChatOpenAI
    from langchain.prompts import ChatPromptTemplate as _ChatPromptTemplate
    ChatOpenAI_impl = _ChatOpenAI
    ChatPromptTemplate_impl = _ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except Exception:
    try:
        from langchain_openai import ChatOpenAI as _ChatOpenAI2
        from langchain.prompts import ChatPromptTemplate as _ChatPromptTemplate2
        ChatOpenAI_impl = _ChatOpenAI2
        ChatPromptTemplate_impl = _ChatPromptTemplate2
        LANGCHAIN_AVAILABLE = True
    except Exception:
        LANGCHAIN_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------
# FFT functions
# ------------------------------
def _infer_read_dtype_and_label(subtype):
    if not subtype:
        return None, "unknown"
    s = subtype.upper()
    if "PCM_U8" in s or "U8" in s or ("8" == s[-1:] and "PCM" in s):
        return "uint8", "8-bit PCM unsigned"
    if "PCM_16" in s or "16" in s:
        return "int16", "16-bit PCM"
    if "PCM_24" in s or "24" in s:
        return "int32", "24-bit PCM"
    if "PCM_32" in s or "32" in s:
        return "int32", "32-bit PCM"
    if "FLOAT" in s or "PCM_FLOAT" in s:
        return "float32", "32-bit float"
    if "DOUBLE" in s or "PCM_DOUBLE" in s:
        return "float64", "64-bit float"
    return None, subtype

def extract_windowed_fft(wav_file, window_size=2048, hop_size=512):
    info = sf.info(wav_file)
    dtype_req, bit_depth_label = _infer_read_dtype_and_label(info.subtype)
    data, rate = sf.read(wav_file, dtype=dtype_req if dtype_req else None, always_2d=False)
    if hasattr(data, "ndim") and data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    n_freqs = window_size // 2
    windows = {
        "rectangular": np.ones(window_size),
        "hanning": np.hanning(window_size),
        "hamming": np.hamming(window_size),
        "blackman": np.blackman(window_size),
    }
    fft_features = {}
    for name, win in windows.items():
        frames = []
        if n < window_size:
            seg = np.zeros(window_size, dtype=data.dtype)
            seg[:n] = data
            frames.append(np.abs(np.fft.fft(seg)[:n_freqs]))
        else:
            for start in range(0, n - window_size + 1, hop_size):
                seg = data[start:start + window_size] * win
                frames.append(np.abs(np.fft.fft(seg)[:n_freqs]))
        fft_features[name] = np.array(frames)
    return rate, window_size, fft_features

def compute_summary(rate, window_size, fft_array):
    if fft_array.size == 0:
        return {"top5": [], "centroid_hz": 0.0, "spread_hz": 0.0}
    mean_spec = fft_array.mean(axis=0)
    bins = np.arange(len(mean_spec))
    freq_axis = bins * (rate / window_size)
    top5_idx = np.argsort(mean_spec)[-5:][::-1]
    top5 = [{"bin": int(idx), "freq_hz": float(freq_axis[idx]), "mag": float(mean_spec[idx])} for idx in top5_idx]
    centroid = float((freq_axis * mean_spec).sum() / mean_spec.sum()) if mean_spec.sum() > 0 else 0.0
    spread = float(np.sqrt(((freq_axis - centroid)**2 * mean_spec).sum() / mean_spec.sum())) if mean_spec.sum() > 0 else 0.0
    return {"top5": top5, "centroid_hz": centroid, "spread_hz": spread}

def save_fft_excel(fft_features, base_name):
    filename = f"{base_name}_fft.xlsx"
    with pd.ExcelWriter(filename) as writer:
        for name, arr in fft_features.items():
            df = pd.DataFrame(arr)
            df.columns = [f"freq_bin_{i}" for i in range(df.shape[1])]
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print("✅ Saved Excel:", filename)

def run_pipeline(wav_path, output_dir=None):
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    output_dir = output_dir or os.path.dirname(wav_path)
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, base_name)
    rate, window_size, fft_features = extract_windowed_fft(wav_path)
    save_fft_excel(fft_features, output_base)
    summaries = {name: compute_summary(rate, window_size, arr) for name, arr in fft_features.items()}
    json_file = f"{output_base}_fft_log.json"
    with open(json_file, "w") as f:
        json.dump({"fft_summary": summaries}, f, indent=2)
    print("✅ Saved JSON log:", json_file)

