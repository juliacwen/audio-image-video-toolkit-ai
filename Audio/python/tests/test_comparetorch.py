"""
 * File: test_comparetorch.py
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-07
 * Test: Run comparetorch_csv.py on WAV-like and spectrum CSVs; verifies mode detection and first-N rows output.
"""

import subprocess
import sys
from pathlib import Path
import csv

def find_script(script_name: str) -> Path:
    root = Path(__file__).resolve().parent
    for parent in [root] + list(root.parents):
        matches = list(parent.rglob(script_name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {script_name} under {root} parents")

SCRIPT = find_script("comparetorch_csv.py")

def write_csv(path, rows):
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)

def run_script(args):
    print(f"\nRunning {SCRIPT} with args {args}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    return result.stdout

def test_wav_compare(tmp_path):
    wav1 = tmp_path / "wav1.csv"
    wav2 = tmp_path / "wav2.csv"
    rows1 = [(i, float(i)) for i in range(100)]
    rows2 = [(i, float(i)+1) for i in range(100)]
    write_csv(wav1, rows1)
    write_csv(wav2, rows2)

    out = run_script([str(wav1), str(wav2), "0", "10"])
    assert "WAV MODE" in out
    assert "first 10" in out
    print("PASSED: WAV compare")

def test_spectrum_compare(tmp_path):
    spec1 = tmp_path / "spec1.csv"
    spec2 = tmp_path / "spec2.csv"
    rows1 = [(i*0.1, float(i)) for i in range(100)]
    rows2 = [(i*0.1, float(i)+0.5) for i in range(100)]
    write_csv(spec1, rows1)
    write_csv(spec2, rows2)

    out = run_script([str(spec1), str(spec2), "0", "10"])
    assert "SPECTRUM MODE" in out
    assert "first 10" in out
    print("PASSED: Spectrum compare")

