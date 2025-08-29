import os
import csv
import subprocess
import math
import pytest

# Path to the compiled wav_freq_csv binary
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "wav_freq_csv")

def write_wav(path, sr=8000, samples=256, freq=1000):
    import wave, struct
    wf = wave.open(path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    for i in range(samples):
        val = int(10000 * math.sin(2 * math.pi * freq * i / sr))
        wf.writeframesraw(struct.pack('<h', val))
    wf.close()

def load_csv(path):
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # skip header
        for row in r:
            vals.append(float(row[1]))
    return vals

@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "rectangular"])
def test_wav_and_spectrum(tmp_path, window):
    wav = tmp_path / "tone.wav"
    csvout = tmp_path / "tone.csv"
    write_wav(str(wav), sr=8000, samples=256, freq=1000)

    # Run wav_freq_csv with the selected window
    subprocess.run([SCRIPT, str(wav), str(csvout), "0", window], check=True)

    # Check sample CSV
    samples = load_csv(csvout)
    assert len(samples) == 256

    # Check spectrum CSV
    specfile = str(csvout).replace(".csv", "_spectrum.csv")
    mags = load_csv(specfile)
    assert len(mags) > 0
    assert any(v != 0.0 for v in mags)

