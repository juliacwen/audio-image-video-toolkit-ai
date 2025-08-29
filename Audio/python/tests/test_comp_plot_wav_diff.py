import os
import math
import subprocess
import pytest
import torch
import torchaudio

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "audio", "comp_plot_wav_diff.py")

def _make_tone_wav(path, encoding, bits_per_sample, sr=16000, freq=440.0, duration=0.25, shift_samples=0):
    n = int(sr * duration)
    t = torch.arange(n + shift_samples, dtype=torch.float32) / sr
    x = torch.sin(2 * math.pi * freq * t)
    if shift_samples > 0:
        x = x[shift_samples:]
    x = x.unsqueeze(0)  # (1, num_samples), mono
    torchaudio.save(path, x, sr, encoding=encoding, bits_per_sample=bits_per_sample)

@pytest.mark.parametrize(
    "encoding,bps",
    [
        ("PCM_S", 16),  # PCM 16-bit
        ("PCM_S", 24),  # PCM 24-bit
        ("PCM_F", 32),  # Float32
    ],
)
def test_comp_plot_wav_diff_formats(tmp_path, encoding, bps):
    a_wav = tmp_path / f"a_{encoding}_{bps}.wav"
    b_wav = tmp_path / f"b_{encoding}_{bps}.wav"
    plot_path = tmp_path / f"plot_{encoding}_{bps}.png"

    # Two slightly different tones to exercise the diff path
    _make_tone_wav(str(a_wav), encoding, bps, shift_samples=0)
    _make_tone_wav(str(b_wav), encoding, bps, shift_samples=5)

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # ensure headless plotting

    # Run the script; pass --save to avoid interactive windows
    subprocess.run(
        [
            "python3",
            SCRIPT,
            str(a_wav),
            str(b_wav),
            "--save",
            str(plot_path),
        ],
        check=True,
        env=env,
    )

    # If the script saved a plot, it should exist and be non-empty.
    if plot_path.exists():
        assert plot_path.stat().st_size > 0
