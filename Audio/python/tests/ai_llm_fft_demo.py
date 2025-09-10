#!/usr/bin/env python3
"""
ai_llm_fft_demo.py - Windowed FFT demo with LLM integration and MLOps-style logging.

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-10
This script:
- Always runs offline FFT (multiple windows), saves Excel (separate sheets), plots average spectra.
- Prints WAV metadata (sampling rate, samples, channels, dtype).
- Reports runtime status for:
    * LangChain (installed / not_installed / installed_but_not_invoked)
    * MLOps (dotenv loaded or missing)
    * LLMOps (OpenAI API key present or missing)
- Attempts an LLM explanation only when LangChain is importable and OPENAI_API_KEY is present.
- Writes MLOps-style JSON log (<basename>_fft_log.json).

MLOps JSON log example:
{
  "runtime_status": {
    "LangChain": "installed",
    "MLOps": "env_loaded",
    "LLMOps": "api_key_missing",
    "reasons": [
      "OPENAI_API_KEY not set (LLM workflow cannot run)",
      "LangChain installed but workflow not invoked (OpenAI key missing)"
    ]
  },
  "fft_summary": {
    "rectangular": {
      "top5": [
        {"bin": 441, "freq_hz": 441.0, "mag": 12345.6}
      ],
      "centroid_hz": 441.0,
      "spread_hz": 20.3
    },
    "hanning": {
      "top5": [
        {"bin": 441, "freq_hz": 441.0, "mag": 11000.5}
      ],
      "centroid_hz": 441.0,
      "spread_hz": 19.7
    },
    "hamming": {
      "top5": [
        {"bin": 441, "freq_hz": 441.0, "mag": 11500.2}
      ],
      "centroid_hz": 441.0,
      "spread_hz": 18.9
    },
    "blackman": {
      "top5": [
        {"bin": 441, "freq_hz": 441.0, "mag": 10500.1}
      ],
      "centroid_hz": 441.0,
      "spread_hz": 17.5
    }
  }
}
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal

# ------------------------------
# Detect optional components
# ------------------------------
DOTENV_LOADED = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_LOADED = True
except Exception:
    DOTENV_LOADED = False

# Try to find a usable ChatOpenAI implementation from LangChain
LANGCHAIN_AVAILABLE = False
ChatOpenAI_impl = None
# Try official langchain.chat_models first, then fallback to langchain_openai if present
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
# We will report statuses later and only call LLM if both LangChain is available and API key present.

# ------------------------------
# FFT extraction & utilities
# ------------------------------
def _infer_read_dtype_and_label(subtype):
    """Map libsndfile subtype to a sensible read dtype and human-readable bit-depth label."""
    if not subtype:
        return None, "unknown"
    s = subtype.upper()
    # common PCM mappings
    if "PCM_U8" in s or "U8" in s or "8" == s[-1:] and "PCM" in s:
        return "uint8", "8-bit PCM unsigned"
    if "PCM_16" in s or "16" in s:
        return "int16", "16-bit PCM"
    if "PCM_24" in s or "24" in s:
        # no native int24 in numpy; use int32 to preserve integer samples
        return "int32", "24-bit PCM"
    if "PCM_32" in s or "32" in s:
        return "int32", "32-bit PCM"
    if "FLOAT" in s or "PCM_FLOAT" in s:
        return "float32", "32-bit float"
    if "DOUBLE" in s or "PCM_DOUBLE" in s:
        return "float64", "64-bit float"
    # fallback: return None (let soundfile choose default dtype)
    return None, subtype

def extract_windowed_fft(wav_file, window_size=2048, hop_size=512):
    # Inspect file to determine subtype / bit depth
    try:
        info = sf.info(wav_file)
        subtype = info.subtype
        samplerate_from_info = info.samplerate
        frames_from_info = info.frames
        channels_from_info = info.channels
    except Exception:
        # If info fails, fall back to naive read (soundfile default)
        subtype = None
        samplerate_from_info = None
        frames_from_info = None
        channels_from_info = None

    dtype_req, bit_depth_label = _infer_read_dtype_and_label(subtype)

    # Try to read with inferred dtype; if that fails, fall back to default read
    try:
        if dtype_req:
            data, rate = sf.read(wav_file, dtype=dtype_req, always_2d=False)
        else:
            data, rate = sf.read(wav_file, always_2d=False)
    except Exception:
        # fallback read (no dtype)
        data, rate = sf.read(wav_file, always_2d=False)

    # If bit_depth_label is unknown, try to fill from data dtype
    if bit_depth_label == "unknown" or bit_depth_label == subtype:
        # give a friendly label if we can
        if hasattr(data, "dtype"):
            bit_depth_label = str(data.dtype)

    # Print WAV metadata (each print on its own line to avoid long single-line f-strings)
    print("\n📄 Audio Info:")
    print("File:", wav_file)
    print("Sampling rate (Hz):", rate)
    print("Samples:", data.shape[0] if hasattr(data, "shape") else len(data))
    print("Channels:", 1 if (hasattr(data, "ndim") and data.ndim == 1) else (data.shape[1] if hasattr(data, "shape") and len(data.shape) > 1 else 
channels_from_info or 1))
    # Show both bit-depth label and the actual numpy dtype so user can see exact native dtype
    print("Bit depth / dtype:", f"{bit_depth_label} / {data.dtype}")

    # Convert to mono if needed
    if hasattr(data, "ndim") and data.ndim > 1:
        # averaging will promote integers to float64 which is fine for FFT ops
        data = data.mean(axis=1)

    n = len(data)
    n_freqs = window_size // 2

    # Window dictionary
    windows = {
        "rectangular": np.ones(window_size),
        "hanning": np.hanning(window_size),
        "hamming": np.hamming(window_size),
        "blackman": np.blackman(window_size),
    }

    fft_features = {}
    for name, win in windows.items():
        frames = []
        # zero-pad if necessary to get at least one frame
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
    top5 = []
    for idx in top5_idx:
        top5.append({"bin": int(idx), "freq_hz": float(freq_axis[idx]), "mag": float(mean_spec[idx])})
    if mean_spec.sum() > 0.0:
        centroid = float((freq_axis * mean_spec).sum() / mean_spec.sum())
        spread = float(np.sqrt(((freq_axis - centroid) ** 2 * mean_spec).sum() / mean_spec.sum()))
    else:
        centroid = 0.0
        spread = 0.0
    return {"top5": top5, "centroid_hz": centroid, "spread_hz": spread}

# ------------------------------
# Save Excel
# ------------------------------
def save_fft_excel(fft_features, base_name):
    filename = f"{base_name}_fft.xlsx"
    try:
        with pd.ExcelWriter(filename) as writer:
            for name, arr in fft_features.items():
                df = pd.DataFrame(arr)
                df.columns = [f"freq_bin_{i}" for i in range(df.shape[1])]
                sheet = name if len(name) <= 31 else name[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
        print("✅ FFT features saved to Excel:", filename)
    except Exception as e:
        print("⚠️ Could not write Excel file:", e)
        print("If you want Excel output, install openpyxl (pip install openpyxl).")

# ------------------------------
# Plot combined average spectra
# ------------------------------
def plot_combined_avg(fft_features, base_name):
    color_map = {"rectangular": "#FF0000", "hanning": "#800080", "hamming": "#FFD700", "blackman": "#008000"}
    style_map = {"rectangular": "-", "hanning": "--", "hamming": "-.", "blackman": ":"}
    plt.figure(figsize=(11, 6))
    for name, arr in fft_features.items():
        if arr.size == 0:
            continue
        avg = arr.mean(axis=0)
        plt.plot(avg, label=name, color=color_map.get(name, "#000000"), linestyle=style_map.get(name, "-"), linewidth=2)
    plt.title("Average FFT Magnitudes Across Window Types")
    plt.xlabel("Frequency bin index")
    plt.ylabel("Magnitude (abs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        plot_file = f"{base_name}_fft_plot.png"
        plt.savefig(plot_file)
        print("Saved plot to", plot_file)

# ------------------------------
# Main pipeline
# ------------------------------
def run_pipeline(wav_path):
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    try:
        rate, window_size, fft_features = extract_windowed_fft(wav_path)
    except Exception as e:
        print("❌ Failed to read or FFT WAV file:", e)
        return

    save_fft_excel(fft_features, base_name)

    summaries = {}
    for name, arr in fft_features.items():
        summaries[name] = compute_summary(rate, window_size, arr)

    print("\nTop-5 average peaks per window (offline):")
    for name, s in summaries.items():
        print(f"\n{name}:")
        if s["top5"]:
            for p in s["top5"]:
                print(f" - {p['freq_hz']:.1f} Hz (bin {p['bin']}) mag={p['mag']:.6g}")
        else:
            print(" - (no data)")
        print(f" - centroid: {s['centroid_hz']:.1f} Hz, spread: {s['spread_hz']:.1f} Hz")

    plot_combined_avg(fft_features, base_name)

    runtime_status = {"LangChain": None, "MLOps": None, "LLMOps": None, "reasons": []}

    if LANGCHAIN_AVAILABLE:
        runtime_status["LangChain"] = "installed"
    else:
        runtime_status["LangChain"] = "not_installed"
        runtime_status["reasons"].append("LangChain not installed (framework unavailable)")

    if DOTENV_LOADED:
        runtime_status["MLOps"] = "env_loaded"
    else:
        runtime_status["MLOps"] = "dotenv_missing"
        runtime_status["reasons"].append("python-dotenv not loaded (environment variables may be missing)")

    if OPENAI_API_KEY:
        runtime_status["LLMOps"] = "api_key_present"
    else:
        runtime_status["LLMOps"] = "api_key_missing"
        runtime_status["reasons"].append("OPENAI_API_KEY not set (LLM workflow cannot run)")

    if LANGCHAIN_AVAILABLE and not OPENAI_API_KEY:
        runtime_status["reasons"].append("LangChain installed but workflow not invoked (OpenAI key missing)")

    print("\nRuntime status (detailed):")
    print("- LangChain:", runtime_status["LangChain"])
    print("- MLOps (dotenv):", runtime_status["MLOps"])
    print("- LLMOps (OpenAI key):", runtime_status["LLMOps"])
    if runtime_status["reasons"]:
        print("\nNotes:")
        for r in runtime_status["reasons"]:
            print("-", r)

    llm_result = None
    llm_notes = []
    if LANGCHAIN_AVAILABLE and OPENAI_API_KEY:
        print("\nℹ️ LangChain and OpenAI key detected — attempting LLM explanation...")
        llm_result, llm_notes = run_llm_explanation_if_available(rate, window_size, summaries)

    if llm_result:
        print("\n✅ LLM Explanation:\n")
        print(llm_result)
    else:
        if llm_notes:
            print("\n⚠️ LLM run notes:")
            for n in llm_notes:
                print("-", n)
        else:
            print("\nℹ️ LLM explanation not produced (see runtime status above).")

    # Always include reasons in JSON
    if "reasons" not in runtime_status:
        runtime_status["reasons"] = []

    log = {"runtime_status": runtime_status, "fft_summary": {}}
    for name, s in summaries.items():
        log["fft_summary"][name] = {
            "top5": s["top5"],
            "centroid_hz": s["centroid_hz"],
            "spread_hz": s["spread_hz"]
        }
   
    log = {"runtime_status": runtime_status, "fft_summary": summaries}
    json_file = f"{base_name}_fft_log.json"
    try:
        with open(json_file, "w") as f:
            json.dump(log, f, indent=2)
        print("\n📝 Workflow log saved to", json_file)
    except Exception as e:
        print("⚠️ Could not write workflow log:", e)



# ------------------------------
# CLI entry
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_fft_demo.py path/to/audio.wav")
        sys.exit(1)
    run_pipeline(sys.argv[1])

