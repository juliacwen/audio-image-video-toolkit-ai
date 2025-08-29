import os
os.environ["MPLBACKEND"] = "Agg"  # force matplotlib to use non-GUI backend

import subprocess
import pathlib
import pandas as pd

samples = pathlib.Path("test_files/samples.csv")
samples5 = pathlib.Path("test_files/samples_5.csv")

def safe_start_and_limit(csv_file, default_limit=200):
    """Pick a safe non-silent start/limit based on file length."""
    df = pd.read_csv(csv_file)
    length = len(df)
    start = length // 2 if length > default_limit else 0
    limit = min(default_limit, length - start)
    return str(start), str(limit)

def run_compare(script, file1, file2, start, limit):
    return subprocess.run(
        ["python3", script, str(file1), str(file2),
         "--start", start, "--limit", limit],
        capture_output=True,
        text=True
    )

def test_compare_csv_runs_identical():
    start, limit = safe_start_and_limit(samples)
    result = run_compare("python/audio/compare_csv.py", samples, samples, start, limit)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr

def test_compare_csv_runs_different():
    start, limit = safe_start_and_limit(samples)
    result = run_compare("python/audio/compare_csv.py", samples, samples5, start, limit)
    assert result.returncode == 0, result.stderr

