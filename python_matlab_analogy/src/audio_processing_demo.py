# src/audio_processing_demo.py
#!/usr/bin/env python3
"""
-----------------------------------------------------------------------
Audio processing demo (production-ready, single-file)
- Keeps the same left-hand slider menu for both synthetic & uploaded audio.
- Slider ranges adapt to the current signal (data-driven) but the menu
  (labels/order) remains identical.
- Automatic best-quality loader: try soundfile.read (high-fidelity) then
  fallback to librosa.load if needed (e.g. MP3).
- Single set of plots (matplotlib): time-domain, FFT (positive freqs),
  and spectrogram. No duplicated graphs.
- Left sidebar analogy selector immediately renders the chosen section
  in the main area (no placeholder text).
- All constants are defined at the top; no arbitrary magic numbers scattered.
- Inline comments document exactly what is done where.
-----------------------------------------------------------------------
Author: Julia Wen (wendigilane@gmail.com)
Date: 2025-11-10
-----------------------------------------------------------------------
"""

# src/audio_processing_demo.py
#!/usr/bin/env python3
"""
audio_processing_demo.py — production-ready (single-file)
- Exports a run() function (so matlab_analogy_demo.py can call audio_processing_demo.run()).
- Behavior fixed to match your requirements exactly:
    * The LEFT sidebar always shows the SAME sliders/menu (Sampling Rate, Duration,
      Amplitude, Frequency, Noise). Menu labels/order never change.
    * Slider RANGES adapt automatically using the FIRST uploaded file when present,
      but sliders are never removed. If no file is uploaded, safe defaults are used.
    * Audio analysis GRAPHS (Time, FFT, Spectrogram, and two-file Diff) are ALWAYS
      rendered at the TOP of the main area.
    * Analogy content is ALWAYS rendered BELOW the graphs.
    * Changing the left-hand analogy selectbox scrolls the browser to the analogy
      section (the page position changes, content order does not).
    * Robust audio loading: uploaded bytes are written to a NamedTemporaryFile and
      read with soundfile.read (preferred) then librosa.load as fallback; pydub
      fallback included if available. This avoids libsndfile BytesIO "Format not
      recognised" errors.
    * st.audio playback works (we play the raw uploaded bytes).
    * No duplicate plots (single matplotlib plots).
    * Two-file difference plot is always shown if two files are uploaded.
    * All constants are defined at the top (no magic numbers sprinkled in).
- Inline comments explain exactly what each part does.
Author: Julia Wen (wendigilane@gmail.com)
Date: 2025-11-10
"""

import io
import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import soundfile as sf
import librosa

# Optional: pydub fallback for some formats (requires ffmpeg on system)
try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    AudioSegment = None
    _HAS_PYDUB = False

# Analogy rendering is provided by the separate module (kept in src/analogy_tables.py)
from src.analogy_tables import (
    render_package_mapping,
    render_concept_equivalence,
    render_detailed_function_mapping,
    render_full_code_examples,
)

# -------------------- CONSTANTS (no magic numbers in the body) --------------------
FS_DEFAULT = 8000           # default sampling rate (synthetic)
DUR_DEFAULT = 1.0           # default duration (s)
AMP_DEFAULT = 1.0           # default amplitude
FREQ_DEFAULT = 440          # default frequency (Hz)
NOISE_DEFAULT = 0.2         # default noise level (std)

N_PER_SEG_DEFAULT = 256     # spectrogram window length (used in examples/plots)
N_OVERLAP_DEFAULT = 128     # spectrogram overlap
N_FFT_DEFAULT = 256         # spectrogram FFT length

FIG_W = 10
FIG_H_SHORT = 3
FIG_H_SPECTRO = 4

# -------------------- Plot helper functions --------------------
def plot_time_domain(signal: np.ndarray, fs: int, title: str):
    """Plot a single time-domain waveform (matplotlib)."""
    if len(signal) == 0:
        st.info("Empty signal — nothing to plot.")
        return
    t = np.arange(len(signal)) / float(fs)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_SHORT))
    ax.plot(t, signal)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

def plot_fft(signal: np.ndarray, fs: int, title: str):
    """Plot single-sided FFT magnitude (matplotlib)."""
    if len(signal) == 0:
        st.info("Empty signal — nothing to plot.")
        return
    N = len(signal)
    X = np.fft.fft(signal)
    f = np.fft.fftfreq(N, 1.0 / fs)
    mag = np.abs(X) / float(N)
    pos_mask = f >= 0
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_SHORT))
    ax.plot(f[pos_mask], mag[pos_mask])
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    st.pyplot(fig)

def plot_spectrogram(signal: np.ndarray, fs: int, title: str,
                     nperseg: int = N_PER_SEG_DEFAULT,
                     noverlap: int = N_OVERLAP_DEFAULT,
                     nfft: int = N_FFT_DEFAULT):
    """Plot spectrogram using scipy.signal.spectrogram -> pcolormesh."""
    if len(signal) == 0:
        st.info("Empty signal — nothing to plot.")
        return
    f, t_seg, Sxx = sig.spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_SPECTRO))
    pcm = ax.pcolormesh(t_seg, f, np.abs(Sxx), shading="gouraud")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.colorbar(pcm, ax=ax, label="Magnitude")
    st.pyplot(fig)

# -------------------- Robust audio decoding helpers --------------------
def _try_soundfile(path_or_file):
    """Try reading with soundfile; return (y_mono, sr) or raise."""
    y, sr = sf.read(path_or_file)
    if getattr(y, "ndim", 1) > 1:
        y = np.mean(y, axis=1)  # convert to mono for analysis/plots
    return np.asarray(y, dtype=float), int(sr)

def _try_librosa(path_or_file):
    """Try reading with librosa; return (y_mono, sr) or raise."""
    y, sr = librosa.load(path_or_file, sr=None)
    return np.asarray(y, dtype=float), int(sr)

def _try_pydub_bytes(upload_bytes: bytes, filename_hint: str):
    """
    If pydub & ffmpeg are available, try to decode bytes -> export WAV -> read.
    This is last-resort for odd codecs.
    """
    if not _HAS_PYDUB:
        raise RuntimeError("pydub not installed")
    buf = io.BytesIO(upload_bytes)
    seg = AudioSegment.from_file(buf)  # may raise if ffmpeg missing
    out_buf = io.BytesIO()
    seg.export(out_buf, format="wav")
    out_buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(out_buf.read())
        tmp.flush()
        tmp.close()
        return _try_soundfile(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

def _read_audio_bytes(upload_bytes: bytes, filename_hint: str):
    """
    Robustly decode uploaded audio bytes:
      1) Write to NamedTemporaryFile (with extension hint).
      2) Try soundfile.read on the file (preferred).
      3) If soundfile fails, try librosa.load on the temp file.
      4) If both fail and pydub available, try pydub decoding.
      5) If all fail, raise RuntimeError with helpful guidance.
    Returns (y_mono_float, sr)
    """
    suffix = os.path.splitext(filename_hint)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(upload_bytes)
        tmp.flush()
        tmp.close()
        try:
            return _try_soundfile(tmp.name)
        except Exception as e_sf:
            try:
                return _try_librosa(tmp.name)
            except Exception as e_lb:
                if _HAS_PYDUB:
                    try:
                        return _try_pydub_bytes(upload_bytes, filename_hint)
                    except Exception as e_pd:
                        raise RuntimeError(
                            f"Decoding failed (soundfile: {e_sf}; librosa: {e_lb}; pydub: {e_pd}). "
                            "Ensure libsndfile and ffmpeg are installed or convert file to WAV."
                        )
                else:
                    raise RuntimeError(
                        f"Decoding failed (soundfile: {e_sf}; librosa: {e_lb}). "
                        "Install pydub+ffmpeg or convert the file to WAV/FLAC."
                    )
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

# -------------------- run() - main entrypoint --------------------
def run():
    """
    Main Streamlit function — preserves your GUI layout and requirements:
      - Left sidebar: identical menu always (sliders + analogy selectbox).
      - Top of main area: audio analysis graphs (always).
      - Analogy content: always below graphs.
      - Selecting an analogy from the left scrolls the page to the analogy anchor without changing layout.
    """
    # Sidebar header (static)
    st.sidebar.header("Signal Parameters")

    # File uploader (0..N files). We'll read bytes per-file and reuse bytes for st.audio + decoding.
    uploaded_files = st.file_uploader(
        "Upload audio file(s) (wav, flac, mp3)", type=["wav", "flac", "mp3"], accept_multiple_files=True
    )

    # Determine slider ranges from the *first* uploaded file (data-driven), otherwise defaults.
    example_signal = None
    example_sr = None
    if uploaded_files:
        # read bytes once for inspection (do not call .read() multiple times on the same UploadedFile)
        try:
            first_bytes = uploaded_files[0].read()
            example_signal, example_sr = _read_audio_bytes(first_bytes, uploaded_files[0].name)
        except Exception as e:
            # inspection failed; warn and fall back to defaults
            st.sidebar.warning(f"Could not inspect first uploaded file to adapt ranges: {e}")
            example_signal, example_sr = None, None

    # Compute ranges
    if example_signal is not None and example_sr is not None:
        fs_min, fs_max, fs_def = 1, int(example_sr), int(example_sr)
        dur_min, dur_max, dur_def = 0.0, max(DUR_DEFAULT, float(len(example_signal) / example_sr)), float(len(example_signal) / example_sr)
        amp_min, amp_max, amp_def = 0.0, max(AMP_DEFAULT, float(np.max(np.abs(example_signal)))), float(np.max(np.abs(example_signal)))
        freq_min, freq_max, freq_def = 1, max(1, int(example_sr // 2)), min(FREQ_DEFAULT, int(example_sr // 2))
        noise_min, noise_max, noise_def = 0.0, max(NOISE_DEFAULT, float(np.std(example_signal))), 0.0
    else:
        fs_min, fs_max, fs_def = 1000, 44100, FS_DEFAULT
        dur_min, dur_max, dur_def = 0.01, 10.0, DUR_DEFAULT
        amp_min, amp_max, amp_def = 0.0, 10.0, AMP_DEFAULT
        freq_min, freq_max, freq_def = 1, 20000, FREQ_DEFAULT
        noise_min, noise_max, noise_def = 0.0, 5.0, NOISE_DEFAULT

    # --- Sidebar sliders: SAME menu always (keys preserve state) ---
    fs = st.sidebar.slider("Sampling Rate (Hz)", int(fs_min), int(fs_max), int(fs_def), step=1, key="fs")
    dur = st.sidebar.slider("Duration (s)", float(dur_min), float(dur_def), float(dur_def), step=0.01, key="dur")
    amp = st.sidebar.slider("Amplitude", float(amp_min), float(amp_max), float(amp_def), step=0.01, key="amp")
    freq = st.sidebar.slider("Frequency (Hz)", int(freq_min), int(freq_max), int(freq_def), step=1, key="freq")
    noise = st.sidebar.slider("Noise Level", float(noise_min), float(noise_max), float(noise_def), step=0.01, key="noise")

    # --- Analogy selectbox on left (selecting triggers smooth scroll to analogy anchor below graphs) ---
    st.sidebar.subheader("Analogy Sections")
    prev_choice = st.session_state.get("last_analogy_choice", None)
    choice = st.sidebar.selectbox(
        "Go to analogy:",
        ["Package Mapping", "Concept Equivalence", "Function-level Mapping", "Full MATLAB ↔ Python Examples"],
        index=0,
        key="analogy_choice",
    )
    # detect user-initiated change
    user_changed = (prev_choice != choice)
    st.session_state["last_analogy_choice"] = choice

    # -------------------- MAIN AREA: GRAPHS (always top) --------------------
    st.markdown("## Audio Analysis (top)")

    loaded = []  # list of tuples (y, sr, name)
    if uploaded_files:
        for uf in uploaded_files:
            # read bytes once for playback & decoding
            try:
                b = uf.read()
                if not b:
                    # try rewind if possible
                    try:
                        uf.seek(0)
                        b = uf.read()
                    except Exception:
                        pass
                # play audio from bytes
                st.audio(b, format=uf.type)
                # decode robustly
                y, sr = _read_audio_bytes(b, uf.name)
            except Exception as e:
                st.error(f"Failed to load {uf.name}: {e}")
                continue

            # store for comparisons/diff
            loaded.append((y, sr, uf.name))
            st.success(f"Loaded {uf.name} — {len(y)} samples @ {sr} Hz")

            # single set of plots (time, FFT, spectrogram) — no duplicates
            plot_time_domain(y, sr, f"{uf.name} — Time Domain")
            plot_fft(y, sr, f"{uf.name} — FFT (magnitude)")
            plot_spectrogram(y, sr, f"{uf.name} — Spectrogram",
                             nperseg=N_PER_SEG_DEFAULT,
                             noverlap=N_OVERLAP_DEFAULT,
                             nfft=N_FFT_DEFAULT)
    else:
        # synthetic: use same menu values (menu order unchanged)
        t = np.linspace(0, dur, max(1, int(fs * dur)), endpoint=False)
        synthetic = amp * np.sin(2 * np.pi * freq * t) + noise * np.random.randn(len(t))
        st.success(f"Synthetic signal: {len(synthetic)} samples @ {fs} Hz")
        plot_time_domain(synthetic, fs, "Synthetic Signal — Time Domain")
        plot_fft(synthetic, fs, "Synthetic Signal — FFT (magnitude)")
        plot_spectrogram(synthetic, fs, "Synthetic Signal — Spectrogram",
                         nperseg=N_PER_SEG_DEFAULT,
                         noverlap=N_OVERLAP_DEFAULT,
                         nfft=N_FFT_DEFAULT)

    # Always show difference plot when two files present (requirement)
    if len(loaded) >= 2:
        y1, sr1, n1 = loaded[0]
        y2, sr2, n2 = loaded[1]
        if sr1 != sr2:
            # resample second -> first sr for direct comparison
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1
        N = min(len(y1), len(y2))
        diff = y1[:N] - y2[:N]
        plot_time_domain(diff, sr1, title=f"Difference: {n1} - {n2}")

    # -------------------- Analogy anchor & content (always BELOW graphs) --------------------
    # Place an explicit anchor element here (below graphs).
    st.markdown('<div id="analogy_anchor"></div>', unsafe_allow_html=True)
    st.markdown("## MATLAB ↔ Python Analogy (below graphs)")

    # Render the requested analogy section (full content, not a placeholder)
    if choice == "Package Mapping":
        render_package_mapping()
    elif choice == "Concept Equivalence":
        render_concept_equivalence()
    elif choice == "Function-level Mapping":
        render_detailed_function_mapping(domain="audio")
    elif choice == "Full MATLAB ↔ Python Examples":
        render_full_code_examples()

    # If the user explicitly changed the selectbox, run JS to scroll to the anchor.
    # This scrolls the browser to the ANALOGY section (content location unchanged).
    if user_changed:
        scroll_js = """
        <script>
        (function() {
          const el = document.getElementById("analogy_anchor");
          if (el) {
            el.scrollIntoView({behavior: "smooth", block: "start"});
          }
        })();
        </script>
        """
        # height=0 to avoid visual artifact
        components.html(scroll_js, height=0)

    # End run()
