import os
import csv
import subprocess
import math

# Point to the compiled binary in project root
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "wav_to_csv")

def write_wav(path, sr=8000, samples=128, freq=440):
    import wave, struct
    wf = wave.open(path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    for i in range(samples):
        val = int(10000 * math.sin(2*math.pi*freq*i/sr))
        wf.writeframesraw(struct.pack('<h', val))
    wf.close()

def load_csv(path):
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            vals.append(float(row[1]))
    return vals

def test_wav_to_csv(tmp_path):
    wav = tmp_path / "tone.wav"
    csvout = tmp_path / "tone.csv"
    write_wav(str(wav))

    subprocess.run([SCRIPT, str(wav), str(csvout)], check=True)
    samples = load_csv(csvout)
    assert len(samples) > 0
    assert any(v != 0.0 for v in samples)
