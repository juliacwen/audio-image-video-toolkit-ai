# python/tests/test_ai_fft_windowing.py
"""
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-12
 * Updated: 2025-09-14
 *
 * Pytest wrapper to run ai_fft_windowing.py
 *
 * Defaults:
 *   - Saves CSV + plot to test_output/
 *   - No interactive display
 *
 * Optional display during pytest:
 *   AI_FFT_SHOW_PLOT=1 pytest -v python/tests/test_ai_fft_windowing.py
 *
 * Manual python3 runs:
 *   python3 python/tests/ai_tools/ai_fft_windowing.py test_files/example.wav --no-plot
 *   python3 python/tests/ai_tools/ai_fft_windowing.py test_files/example.wav --show-plot
 *   python3 python/tests/ai_tools/ai_fft_windowing.py test_files/example.wav --outdir my_output
"""

import subprocess
from pathlib import Path
import pytest
import os

# Collect all WAV files in test_files/
wav_dir = Path("test_files")
wav_files = list(wav_dir.glob("*.wav"))

@pytest.mark.parametrize("wavfile", wav_files, ids=[f.name for f in wav_files])
def test_ai_fft_windowing(wavfile):
    """Run ai_fft_windowing.py on each WAV file in test_files/."""
    script = Path("python/tests/ai_tools/ai_fft_windowing.py")
    assert script.exists(), f"Script not found: {script}"
    assert wavfile.exists(), f"WAV file not found: {wavfile}"

    env = os.environ.copy()  # optionally set AI_FFT_SHOW_PLOT=1 externally for display

    ret = subprocess.run(
        ["python3", str(script), str(wavfile), "--outdir", "test_output"],
        capture_output=True,
        text=True,
        env=env
    )

    assert ret.returncode == 0, (
        f"Script failed with code {ret.returncode} for {wavfile.name}\n"
        f"stdout:\n{ret.stdout}\n"
        f"stderr:\n{ret.stderr}"
    )

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-v", __file__]))

