"""
 * File: test_compare_csv.py
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-07
 * Test: Run compare_csv.py on identical and slightly different CSV files; verifies that
 *       return codes are 0 and no traceback occurs.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

os.environ["MPLBACKEND"] = "Agg"  # force matplotlib to use non-GUI backend

def find_script(script_name: str) -> Path:
    root = Path(__file__).resolve().parent
    for parent in [root] + list(root.parents):
        matches = list(parent.rglob(script_name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {script_name} under {root} parents")

SCRIPT = find_script("compare_csv.py")

BASE = Path(__file__).resolve().parent
samples = BASE / "test_files" / "samples.csv"
samples5 = BASE / "test_files" / "samples_5.csv"

def safe_start_and_limit(csv_file, default_limit=200):
    """Pick a safe non-empty start/limit based on file length."""
    df = pd.read_csv(csv_file)
    length = len(df)
    start = length // 2 if length > default_limit else 0
    limit = min(default_limit, length - start)
    return str(start), str(limit)

def run_compare(script, file1, file2, start, limit):
    print(f"\nRunning {script} with {file1} vs {file2}, start={start}, limit={limit}")
    result = subprocess.run(
        [sys.executable, str(script), str(file1), str(file2), "--start", start, "--limit", limit],
        capture_output=True,
        text=True,
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    return result

def test_compare_csv_runs_identical():
    start, limit = safe_start_and_limit(samples)
    result = run_compare(SCRIPT, samples, samples, start, limit)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    print("PASSED: compare_csv identical files")

def test_compare_csv_runs_different():
    start, limit = safe_start_and_limit(samples)
    result = run_compare(SCRIPT, samples, samples5, start, limit)
    assert result.returncode == 0, result.stderr
    print("PASSED: compare_csv different files")

