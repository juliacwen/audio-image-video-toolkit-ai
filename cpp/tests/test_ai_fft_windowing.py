import subprocess
from pathlib import Path

def test_ai_fft_windowing():
    """
    Pytest wrapper to run full AI neural network spectrum analysis workflow.
    """
    script = Path(__file__).parent / "ai_tools" / "ai_fft_workflow.py"
    ret = subprocess.run(["python3", str(script)])
    assert ret.returncode == 0

