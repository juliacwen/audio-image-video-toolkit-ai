#!/usr/bin/env python3
#Audio/bayesian/app_bayesian_fft_streamlit_db.py

"""
app_bayesian_fft_streamlit_db.py
Author: Julia Wen (wendigilane@gmail.com)
Date: 2025-10-31
Description:
------------
Streamlit web application for Bayesian Monte Carlo FFT analysis with uncertainty estimation.
Stores results directly in `multimodal.db` (sqlite) at project root.

Key Features:
-------------
1. AI/ML Integration:
   - Uses PyTorch tensors for efficient numerical computation of FFT.
   - Optional GPU acceleration for faster Monte Carlo simulations.
   - Handles large audio signals efficiently using tensor operations.

2. Bayesian Monte Carlo FFT:
   - Adds controlled noise to generate multiple FFT samples.
   - Computes posterior distribution of frequency peaks.
   - Reports mean and standard deviation of detected peaks for uncertainty quantification.

3. Audio Support:
   - Reads WAV, AIFF, FLAC, MP3, M4A formats using `soundfile` and `pydub`.
   - Processes first channel for multi-channel audio.
   - Automatically identifies first non-silent segment for zoomed analysis.

4. Streamlit Interactive UI:
   - Select single or pair of audio files for comparison.
   - Adjustable Monte Carlo sample count and noise level.
   - Rough and precise time-domain zoom sliders.
   - Interactive plots: time-domain waveform, FFT, and posterior distribution histogram.

5. Database Integration:
   - Automatically creates/updates `multimodal.db` (sqllite) in the project root.
   - Stores FFT results per file: frequency, FFT amplitude, Monte Carlo sample peaks, mean, and std.
   - Works with downstream export utilities to convert database entries to CSV if needed.

6. Visualization:
   - Full-signal FFT plot with mean peak markers.
   - Posterior histogram of Monte Carlo samples.
   - Overlay plots for comparing multiple files.

Usage:
------
Run the app from the project root:

    streamlit run bayesian/app_bayesian_fft_streamlit_db.py

This DB version is for users who prefer centralized storage for batch processing,
analytics, or integration with other tools, while still allowing CSV export if desired.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf
from pydub import AudioSegment
import streamlit as st
import sqlite3
import pandas as pd

# --- Configuration ---
DEFAULT_MONTE_CARLO_SAMPLES = 200
DEFAULT_NOISE_LEVEL = 0.01
PLOT_FIGSIZE = (8, 4)
HIST_BINS = 50

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- File paths ---
BASE_DIR = os.path.dirname(__file__)
TEST_FILES_DIR = os.path.join(BASE_DIR, "../../test_files")
os.makedirs(TEST_FILES_DIR, exist_ok=True)

# --- Database path: always project root ---
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # two levels up from bayesian
DB_FILE = os.path.join(PROJECT_ROOT, "multimodal.db")

# Ensure database exists and create table if missing
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bayesian_fft (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            freq REAL,
            fft_val REAL,
            sample_peak REAL,
            mean_peak REAL,
            std_peak REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()  # create DB and table if missing

# --- Bayesian FFT ---
def bayesian_fft(signal, fs, n_samples, noise_level):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    sampled_peaks = []

    sig_tensor = torch.tensor(signal, dtype=torch.float32, device=device)

    for _ in range(n_samples):
        noisy_sig = sig_tensor + noise_level * torch.randn(N, device=device)
        fft_sample = torch.fft.rfft(noisy_sig)
        fft_abs = torch.abs(fft_sample)
        peak_idx = torch.argmax(fft_abs).item()
        sampled_peaks.append(freqs[peak_idx])

    return freqs, fft_vals, np.array(sampled_peaks)

# --- Load audio ---
def load_audio_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".wav", ".flac", ".aiff", ".aif"]:
        sig, fs = sf.read(path, dtype='float32')
        if sig.ndim > 1:
            sig = sig[:, 0]
        return sig, fs
    else:
        audio = AudioSegment.from_file(path)
        fs = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))[:, 0]
        max_val = float(2 ** (8 * audio.sample_width - 1))
        signal = samples / max_val
        return signal, fs

# --- First non-silent portion ---
def first_non_silent(signal, threshold=0.01, min_len=1000):
    abs_sig = np.abs(signal)
    above_thr = np.where(abs_sig > threshold)[0]
    if len(above_thr) == 0:
        return 0, len(signal)
    start = above_thr[0]
    end = min(start + min_len, len(signal))
    return start, end

# --- Save FFT to database ---
def save_fft_to_db(file_name, freqs, fft_vals, samples, mean_peak, std_peak):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for i in range(len(freqs)):
        cursor.execute("""
            INSERT INTO bayesian_fft (file_name, freq, fft_val, sample_peak, mean_peak, std_peak)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_name, float(freqs[i]), float(fft_vals[i]),
              float(samples[i] if i < len(samples) else 0.0),
              float(mean_peak), float(std_peak)))
    conn.commit()
    conn.close()

# --- Optional: retrieve from db ---
def load_fft_from_db(file_name=None):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM bayesian_fft"
    params = ()
    if file_name:
        query += " WHERE file_name = ?"
        params = (file_name,)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# --- Streamlit UI ---
st.title("Bayesian FFT Demo")
st.write(f"Running on: {device}")
st.write(f"Database path: {DB_FILE}")
st.write("Compute Bayesian FFT with Monte Carlo uncertainty on WAV, AIFF, M4A, or MP3 files.")

supported_ext = [".wav", ".flac", ".aiff", ".aif", ".m4a", ".mp3"]
audio_files = [f for f in os.listdir(TEST_FILES_DIR) if os.path.splitext(f)[1].lower() in supported_ext]

if not audio_files:
    st.warning(f"No supported audio files found in {TEST_FILES_DIR}.")
else:
    selected_file1 = st.selectbox("Select first audio file", audio_files)
    compare_mode = st.checkbox("Compare with a second file?")
    selected_file2 = None
    if compare_mode:
        remaining_files = [f for f in audio_files if f != selected_file1]
        selected_file2 = st.selectbox("Select second audio file", remaining_files)

    # --- Parameters ---
    n_samples = st.number_input("Monte Carlo samples", min_value=10, max_value=5000,
                                value=DEFAULT_MONTE_CARLO_SAMPLES, step=10)
    noise_level = st.number_input("Monte Carlo noise level", min_value=0.0, max_value=1.0,
                                  value=DEFAULT_NOISE_LEVEL, step=0.01, format="%.3f")

    LINE_COLOR1, LINE_COLOR2 = 'blue', 'orange'
    DASHED_COLOR1, DASHED_COLOR2 = 'red', 'green'

    # --- Process files ---
    def process_file_store_state(file_name, key_prefix):
        path = os.path.join(TEST_FILES_DIR, file_name)
        sig, fs = load_audio_any(path)
        freqs, fft_vals, samples = bayesian_fft(sig, fs, n_samples, noise_level)
        mean_peak = np.mean(samples)
        std_peak = np.std(samples)

        st.session_state[f"{key_prefix}_sig"] = sig
        st.session_state[f"{key_prefix}_fs"] = fs
        st.session_state[f"{key_prefix}_freqs"] = freqs
        st.session_state[f"{key_prefix}_fft_vals"] = fft_vals
        st.session_state[f"{key_prefix}_samples"] = samples
        st.session_state[f"{key_prefix}_mean_peak"] = mean_peak
        st.session_state[f"{key_prefix}_std_peak"] = std_peak

        # --- Save to database ---
        save_fft_to_db(file_name, freqs, fft_vals, samples, mean_peak, std_peak)

    if st.button("Run Bayesian FFT") or "sig1" not in st.session_state:
        process_file_store_state(selected_file1, "sig1")
        if compare_mode and selected_file2:
            process_file_store_state(selected_file2, "sig2")

    # --- Retrieve from session_state ---
    sig1 = st.session_state["sig1_sig"]
    fs1 = st.session_state["sig1_fs"]
    freqs1 = st.session_state["sig1_freqs"]
    fft_vals1 = st.session_state["sig1_fft_vals"]
    samples1 = st.session_state["sig1_samples"]
    mean_peak1 = st.session_state["sig1_mean_peak"]
    std_peak1 = st.session_state["sig1_std_peak"]

    if compare_mode and selected_file2:
        sig2 = st.session_state["sig2_sig"]
        fs2 = st.session_state["sig2_fs"]
        freqs2 = st.session_state["sig2_freqs"]
        fft_vals2 = st.session_state["sig2_fft_vals"]
        samples2 = st.session_state["sig2_samples"]
        mean_peak2 = st.session_state["sig2_mean_peak"]
        std_peak2 = st.session_state["sig2_std_peak"]
    else:
        sig2 = fs2 = freqs2 = fft_vals2 = samples2 = mean_peak2 = std_peak2 = None

    st.write(f"**{selected_file1}: mean = {mean_peak1:.2f} Hz, std = {std_peak1:.2f} Hz**")
    if compare_mode and selected_file2:
        st.write(f"**{selected_file2}: mean = {mean_peak2:.2f} Hz, std = {std_peak2:.2f} Hz**")

    # --- Time zoom slider ---
    default_start, default_end = first_non_silent(sig1)
    slider_start, slider_end = st.slider("Rough time zoom (samples)", 0, len(sig1),
                                         value=(default_start, default_end))

    zoom_start = st.number_input("Zoom start sample (precise)", min_value=slider_start,
                                 max_value=slider_end, value=slider_start, step=1)
    zoom_end = st.number_input("Zoom end sample (precise)", min_value=zoom_start+1,
                               max_value=slider_end, value=slider_end, step=1)

    t1_zoomed = np.arange(zoom_start, zoom_end) / fs1
    sig1_zoomed = sig1[zoom_start:zoom_end]

    if sig2 is not None:
        zoom_end2 = min(zoom_end, len(sig2))
        t2_zoomed = np.arange(zoom_start, zoom_end2) / fs2
        sig2_zoomed = sig2[zoom_start:zoom_end2]

    # --- Time-domain waveform ---
    fig_time, ax_time = plt.subplots(figsize=PLOT_FIGSIZE)
    ax_time.plot(t1_zoomed, sig1_zoomed, color='blue', label=f"{selected_file1} waveform")
    if sig2 is not None:
        ax_time.plot(t2_zoomed, sig2_zoomed, color='orange', label=f"{selected_file2} waveform")
    ax_time.axvline(x=zoom_start/fs1, color='red', linestyle='--', linewidth=1)
    ax_time.axvline(x=zoom_end/fs1, color='red', linestyle='--', linewidth=1)
    ax_time.set_xlim(t1_zoomed[0], t1_zoomed[-1])
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Time-domain Waveform (Zoomed)")
    ax_time.legend()
    fig_time.tight_layout()
    st.pyplot(fig_time)

    # --- FFT plot ---
    fig_fft, ax_fft = plt.subplots(figsize=PLOT_FIGSIZE)
    ax_fft.plot(freqs1, fft_vals1, label=f"{selected_file1} FFT", color='blue')
    ax_fft.axvline(mean_peak1, color='red', linestyle='--', label=f"{selected_file1} mean peak")
    if compare_mode and freqs2 is not None:
        ax_fft.plot(freqs2, fft_vals2, label=f"{selected_file2} FFT", color='orange')
        ax_fft.axvline(mean_peak2, color='green', linestyle='--', label=f"{selected_file2} mean peak")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_title("FFT (Full Signal)")
    ax_fft.legend()
    fig_fft.tight_layout()
    st.pyplot(fig_fft)

    # --- Posterior histogram ---
    fig_hist, ax_hist = plt.subplots(figsize=PLOT_FIGSIZE)
    ax_hist.hist(samples1, bins=HIST_BINS, color='blue', alpha=0.5, edgecolor='black',
                 label=f"{selected_file1} samples")
    ax_hist.axvline(mean_peak1, color='red', linestyle='--', label=f"{selected_file1} mean peak")
    if compare_mode and samples2 is not None:
        ax_hist.hist(samples2, bins=HIST_BINS, color='orange', alpha=0.5, edgecolor='black',
                     label=f"{selected_file2} samples")
        ax_hist.axvline(mean_peak2, color='green', linestyle='--', label=f"{selected_file2} mean peak")
    ax_hist.set_xlabel("Frequency (Hz)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Posterior Distribution (Full Signal)")
    ax_hist.legend()
    fig_hist.tight_layout()
    st.pyplot(fig_hist)
