# python/tests/test_ai_fft_windowing.py
"""
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-12

Pytest wrapper to run ai_fft_windowing.py

Commands:

# Run all WAVs in test_files/ with default output (test_output/)
pytest -v python/tests/test_ai_fft_windowing.py

# Run ai_fft_windowing.py manually (all WAVs)
python3 python/tests/ai_tools/ai_fft_windowing.py

# Optional overrides:
# Change output directory
python3 python/tests/ai_tools/ai_fft_windowing.py --outdir my_output_dir
# Disable plotting
python3 python/tests/ai_tools/ai_fft_windowing.py --no-plot
"""
import subprocess
from pathlib import Path
import pytest

# Collect all WAV files in test_files/
wav_dir = Path("test_files")
wav_files = list(wav_dir.glob("*.wav"))

@pytest.mark.parametrize("wavfile", wav_files)
def test_ai_fft_windowing(wavfile):
    """Run ai_fft_windowing.py on each WAV file in test_files/."""
    script = Path("python/tests/ai_tools/ai_fft_windowing.py")

    assert script.exists(), f"Script not found: {script}"
    assert wavfile.exists(), f"WAV file not found: {wavfile}"

    ret = subprocess.run(["python3", str(script), str(wavfile)])
    assert ret.returncode == 0, f"Script failed with code {ret.returncode} for {wavfile}"

