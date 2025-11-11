# src/analogy_tables.py
#!/usr/bin/env python3
"""
analogy_tables.py 
Author: Julia Wen (wendigilane@gmail.com)
Date: 11-10-2025
Description:
— MATLAB ↔ Python analogy tables & code examples
- Includes package-level mapping, concept equivalence, function-level mapping,
  and MATLAB ⇄ Python example code blocks.
"""

import streamlit as st

# -------------------- Constants used in examples (NO magic numbers inline) --------------------
FS_EX = 8000
DUR_EX = 1.0
AMP_EX = 1.0
FREQ_EX = 440
NOISE_EX = 0.2
N_PER_SEG_EX = 256
N_OVERLAP_EX = 128
N_FFT_EX = 256

# -------------------- Package-level mapping --------------------
def render_package_mapping():
    st.markdown("### Package-Level Mapping (MATLAB ↔ Python)")
    st.markdown(
        """
| MATLAB Toolbox / Feature | Python Package(s) | When to use / Notes |
|--------------------------|-------------------|---------------------|
| Signal Processing Toolbox | numpy, scipy.signal | FFT, filtering, STFT/spectrogram — direct equivalents |
| Core MATLAB (array ops)  | numpy               | Vectorized math: sin, randn, abs, angle |
| Audio I/O                | soundfile (preferred), librosa (fallback) | soundfile.read preserves PCM (WAV/FLAC); librosa.load handles MP3/others and normalizes floats. |
| Plotting                 | matplotlib, plotly  | matplotlib for static; plotly for interactive exploration |
| GUI / App Designer       | streamlit           | sliders, file_uploader, etc. (App Designer analogue) |
| Real-time I/O            | sounddevice, pyaudio | For recording/playback (MATLAB audiorecorder/sound analogues) |
| Audio manipulation       | pydub               | Slicing/concatenation/format conversion (use ffmpeg) |
| Deep-learning audio      | torchaudio          | For ML pipelines; analogous to MATLAB Deep Learning Toolbox usage |
        """
    )

# -------------------- Concept equivalence --------------------
def render_concept_equivalence():
    st.markdown("### Concept Equivalence (MATLAB ⇄ Python)")
    st.markdown(
        """
| MATLAB Concept | Python Equivalent | Notes |
|----------------|-------------------|-------|
| sin, randn, array math | numpy.sin, numpy.random.randn | Vectorized operations with broadcasting |
| fft, ifft, abs, angle | numpy.fft.fft/ifft, numpy.abs, numpy.angle | Same transforms and operations |
| audioread, sound | soundfile.read (WAV/FLAC) / librosa.load (MP3) / st.audio | soundfile preserves integer PCM; librosa normalizes to floats |
| butter, filtfilt | scipy.signal.butter, scipy.signal.filtfilt | Filter design + zero-phase filtering |
| spectrogram | scipy.signal.spectrogram, librosa.stft | Visualization via plt.pcolormesh or plotly Heatmap (imagesc analogue) |
| plot, subplot | matplotlib.pyplot.plot, plt.subplots | Plotly for interactive variants |
| UI controls | streamlit sliders, file_uploader | App Designer analogues for demos |
        """
    )

# -------------------- Function-level mapping --------------------
def render_detailed_function_mapping(domain="audio"):
    st.markdown("### Function-Level Mapping")
    if domain == "audio":
        st.markdown(
            """
| MATLAB Function | Python Equivalent | Fidelity / Usage Notes |
|-----------------|-------------------|------------------------|
| audioread(file) | soundfile.read(file) (preferred) / librosa.load(file) | soundfile preserves PCM for WAV/FLAC; librosa supports many formats and returns floats in [-1,1]. |
| sound(x, fs)    | st.audio(bytes) or sounddevice.play(x, fs) | st.audio for demos; sounddevice for real-time playback. |
| fft(x) / ifft(X)| numpy.fft.fft / numpy.fft.ifft | Identical complex FFT operations. |
| abs(X), angle(X)| numpy.abs, numpy.angle | Magnitude & phase extraction. |
| butter, filtfilt | scipy.signal.butter, scipy.signal.filtfilt | Filter design & zero-phase filtering equivalent. |
| spectrogram     | scipy.signal.spectrogram / librosa.stft | Use plt.pcolormesh / plotly Heatmap for visualization (imagesc analogue). |
            """
        )
    else:
        st.info("No detailed mapping for domain: " + str(domain))

# -------------------- Full MATLAB ⇄ Python code examples --------------------
def render_full_code_examples():
    st.markdown("### Full MATLAB ⇄ Python Example Code (constants referenced)")

    st.markdown("**Synthetic sinusoid + noise**")
    st.code(
        f"""% MATLAB
fs = {FS_EX};
t = 0:1/fs:{DUR_EX};
x = sin(2*pi*{FREQ_EX}*t) + {NOISE_EX}*randn(size(t));
plot(t, x);
""",
        language="matlab",
    )
    st.code(
        f"""# Python
import numpy as np
import matplotlib.pyplot as plt
fs = {FS_EX}
T = {DUR_EX}
amp, freq, noise = {AMP_EX}, {FREQ_EX}, {NOISE_EX}
t = np.linspace(0, T, int(fs*T), endpoint=False)
signal = amp*np.sin(2*np.pi*freq*t) + noise*np.random.randn(len(t))
plt.plot(t, signal)
plt.show()
""",
        language="python",
    )
    st.markdown("**Note:** constants (FS_EX etc.) are defined at top of this file to avoid magic numbers.")

    st.markdown("**FFT magnitude (single-sided)**")
    st.code(
        """% MATLAB
X = fft(x);
f = (0:length(X)-1)*(fs/length(X));
plot(f, abs(X)/length(X));
""",
        language="matlab",
    )
    st.code(
        """# Python
X = np.fft.fft(signal)
f = np.fft.fftfreq(len(signal), 1.0/fs)
plt.plot(f[f>=0], np.abs(X[f>=0])/len(signal))
plt.show()
""",
        language="python",
    )

    st.markdown("**Butterworth + filtfilt (zero-phase)**")
    st.code(
        """% MATLAB
[b,a] = butter(4, 0.3, 'low');
y = filtfilt(b,a,x);
plot(y);
""",
        language="matlab",
    )
    st.code(
        """# Python
from scipy import signal as sig
b, a = sig.butter(4, 0.3, 'low')
y = sig.filtfilt(b, a, signal)
plt.plot(y)
plt.show()
""",
        language="python",
    )

    st.markdown("**Spectrogram / STFT (constants used)**")
    st.code(
        f"""% MATLAB
[S,F,T] = spectrogram(x, {N_PER_SEG_EX}, {N_OVERLAP_EX}, {N_FFT_EX}, fs);
imagesc(T, F, abs(S));
xlabel('Time [s]'); ylabel('Frequency [Hz]');
""",
        language="matlab",
    )
    st.code(
        f"""# Python
f, t, Sxx = sig.spectrogram(signal, fs, nperseg={N_PER_SEG_EX}, noverlap={N_OVERLAP_EX}, nfft={N_FFT_EX})
plt.pcolormesh(t, f, np.abs(Sxx))
plt.xlabel('Time [s]'); plt.ylabel('Frequency [Hz]')
plt.show()
""",
        language="python",
    )

    st.markdown(
        """
**Fidelity notes:**  
- `soundfile.read` ≈ MATLAB `audioread` for WAV/FLAC and preserves integer PCM.  
- `librosa.load` is more flexible (MP3 etc.) but normalizes to floating point in [-1,1].  
- `pydub` + ffmpeg can decode many codecs if installed; it's provided here as a last-resort fallback.
"""
    )
