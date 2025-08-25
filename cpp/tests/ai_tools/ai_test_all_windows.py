#!/usr/bin/env python3
import subprocess
from pathlib import Path

# Paths
AI_TOOLS = Path(__file__).parent
OUTPUT_DIR = AI_TOOLS / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Signal parameters
FREQ = 1000        # Hz
SR = 8000          # Sample rate
DURATION_SEC = 1.0 # seconds

WINDOWS = [
    "rectangular",
    "hann",
    "hamming",
    "blackman",
    "bartlett",
]

def generate_wav(window_name: str):
    """
    Generate a test wav file using generate_wav.py
    """
    out_file = OUTPUT_DIR / f"tone_{window_name}.wav"
    subprocess.run([
        "python3", str(AI_TOOLS / "generate_wav.py"),
        str(out_file),
        "--freq", str(FREQ),
        "--sr", str(SR),
        "--duration", str(DURATION_SEC)
    ], check=True)

def run_wav_to_csv(window_name: str):
    """
    Convert generated wav to CSV using wav_freq_csv binary at repo root.
    """
    in_file = OUTPUT_DIR / f"tone_{window_name}.wav"
    out_file = OUTPUT_DIR / f"tone_{window_name}.csv"

    # repo root (3 levels up from this script: cpp/tests/ai_tools → repo/)
    repo_root = Path(__file__).resolve().parents[3]
    binary = repo_root / "wav_freq_csv"

    if not binary.exists():
        raise FileNotFoundError(f"wav_freq_csv not found at {binary}")

    cmd = [
        str(binary),
        str(in_file),
        str(out_file),
        "--window", window_name,
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    for win in WINDOWS:
        print(f"[INFO] Testing window: {win}")
        generate_wav(win)
        run_wav_to_csv(win)
    print("[SUCCESS] All window tests completed.")

if __name__ == "__main__":
    main()

