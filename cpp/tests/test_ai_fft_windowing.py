import subprocess
from pathlib import Path

def test_ai_fft_windowing():
    """
    Pytest wrapper to run AI-assisted FFT windowing test.
    Executes ai_test_all_windows.py and asserts successful completion.
    """
    script = Path(__file__).parent / "ai_tools" / "ai_test_all_windows.py"
    ret = subprocess.run(["python3", str(script)])
    assert ret.returncode == 0

