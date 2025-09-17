#!/usr/bin/env python3
"""
ai_llm_fft_demo.py - Windowed FFT demo with LLM integration and MLOps-style logging.

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-17
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
ChatPromptTemplate_impl = None
try:
    from langchain.chat_models import ChatOpenAI as _ChatOpenAI
    from langchain.prompts import ChatPromptTemplate as _ChatPromptTemplate
    ChatOpenAI_impl = _ChatOpenAI
    ChatPromptTemplate_impl = _ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------
# Device detection
# ------------------------------
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

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
    
    # Print WAV metadata
    n_samples = len(data)
    n_channels = 1 if not hasattr(data, "ndim") or data.ndim == 1 else data.shape[1]
    print(f"📄 WAV metadata: rate={rate} Hz, samples={n_samples}, channels={n_channels}, dtype={data.dtype}, subtype={info.subtype}")

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

def plot_average_spectra(rate, window_size, fft_features, base_name, show_plot=False):
    plt.figure(figsize=(10,6))
    for name, arr in fft_features.items():
        mean_spec = arr.mean(axis=0)
        freq_axis = np.arange(len(mean_spec)) * (rate / window_size)
        plt.plot(freq_axis, mean_spec, label=name)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Average FFT Spectrum")
    plt.legend()
    png_file = f"{base_name}_fft.png"
    plt.savefig(png_file)
    print("✅ Saved PNG plot:", png_file)
    if show_plot:
        plt.show()
    plt.close()

# ------------------------------
# LLM stub
# ------------------------------
def run_llm_explanation_if_available(rate, window_size, summaries):
    if LANGCHAIN_AVAILABLE and OPENAI_API_KEY:
        return {"message": "LLM explanation stub"}, True
    return None, False

# ------------------------------
# Main pipeline
# ------------------------------
def run_pipeline(wav_path, output_dir=None, show_plot=False, verbose=False):
    base_name_only = os.path.splitext(os.path.basename(wav_path))[0]
    output_dir = output_dir or os.path.dirname(wav_path)
    os.makedirs(output_dir, exist_ok=True)

    # Full paths for outputs
    output_base = os.path.join(output_dir, base_name_only)
    excel_file = f"{output_base}_fft.xlsx"
    png_file = f"{output_base}_fft.png"
    json_file = f"{output_base}_fft_log.json"

    if verbose:
        print("🛠 Runtime status:")
        print(f"  - dotenv loaded: {DOTENV_LOADED}")
        print(f"  - LangChain available: {LANGCHAIN_AVAILABLE}")
        print(f"  - OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")
        print(f"  - Device: {DEVICE}")
        print(f"🎵 Processing WAV file: {wav_path}")

    rate, window_size, fft_features = extract_windowed_fft(wav_path)

    # Save Excel
    save_fft_excel(fft_features, output_base)
    
    # Save PNG
    plot_average_spectra(rate, window_size, fft_features, output_base, show_plot=show_plot)

    # Save JSON
    summaries = {name: compute_summary(rate, window_size, arr) for name, arr in fft_features.items()}
    with open(json_file, "w") as f:
        json.dump({"fft_summary": summaries}, f, indent=2)
    print("✅ Saved JSON log:", json_file)

    # Console summary table
    print("\n📊 Average FFT summary (top 5 frequencies per window type):")
    for name, summary in summaries.items():
        print(f"\n-- {name} window --")
        print(f"Centroid: {summary['centroid_hz']:.1f} Hz, Spread: {summary['spread_hz']:.1f} Hz")
        print("Top 5 frequencies (Hz) | Magnitude")
        for item in summary['top5']:
            print(f"  {item['freq_hz']:.1f} Hz\t| {item['mag']:.3f}")

    print("\n🎉 Pipeline finished!")

# ------------------------------
# CLI entry point
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Windowed FFT demo with optional LLM logging")
    parser.add_argument("wav_file", help="Path to input WAV file")
    parser.add_argument("-o", "--output_dir", default=None, help="Directory to save outputs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print runtime info")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    run_pipeline(args.wav_file, args.output_dir, show_plot=args.show, verbose=args.verbose)

