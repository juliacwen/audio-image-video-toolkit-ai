import os
import csv
import subprocess

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "audio", "comparetorch_csv.py")

def write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)

def run_script(args):
    return subprocess.run(
        ["python3", SCRIPT] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    ).stdout

def test_wav_compare(tmp_path):
    # WAV format: Index,Sample
    wav1 = tmp_path / "wav1.csv"
    wav2 = tmp_path / "wav2.csv"
    rows1 = [(i, float(i)) for i in range(100)]
    rows2 = [(i, float(i)+1) for i in range(100)]
    write_csv(wav1, rows1)
    write_csv(wav2, rows2)

    out = run_script([str(wav1), str(wav2), "0", "10"])
    assert "WAV MODE" in out
    assert "first 10" in out

def test_spectrum_compare(tmp_path):
    # Spectrum format: Frequency,Magnitude
    spec1 = tmp_path / "spec1.csv"
    spec2 = tmp_path / "spec2.csv"
    rows1 = [(i*0.1, float(i)) for i in range(100)]
    rows2 = [(i*0.1, float(i)+0.5) for i in range(100)]
    write_csv(spec1, rows1)
    write_csv(spec2, rows2)

    out = run_script([str(spec1), str(spec2), "0", "10"])
    assert "SPECTRUM MODE" in out
    assert "first 10" in out
