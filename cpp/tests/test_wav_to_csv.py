# cpp/tests/test_wav_to_csv.py
import subprocess
import pandas as pd
import numpy as np
import wave
import struct
from pathlib import Path
import pytest


def write_float_wav(path: Path, samples: list[float]):
    """Write IEEE float32 WAV manually (since Python wave only supports PCM)."""
    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 32
    audio_format = 3  # IEEE float
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    subchunk2_size = len(samples) * block_align
    chunk_size = 36 + subchunk2_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", chunk_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))   # PCM fmt chunk size
        f.write(struct.pack("<H", audio_format))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", subchunk2_size))
        for s in samples:
            f.write(struct.pack("<f", s))


def write_test_wav(path: Path, bit_depth: int):
    """Generate test WAV with both zero and non-zero samples."""
    samples = [0, 1000, -1000, 0, 32767, -32768, 0]

    if bit_depth == 16:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            frames = struct.pack("<" + "h" * len(samples), *samples)
            wf.writeframes(frames)
        return np.array(samples, dtype=np.float32), 16

    elif bit_depth == 24:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(16000)
            frames = b""
            for s in samples:
                frames += struct.pack("<i", s)[0:3]  # take low 3 bytes
            wf.writeframes(frames)
        return np.array(samples, dtype=np.float32), 24

    elif bit_depth == 32:
        floats = [float(s) / 32768.0 for s in samples]
        write_float_wav(path, floats)
        return np.array(floats, dtype=np.float32), 32

    else:
        raise ValueError("Unsupported bit depth")


def load_csv(csv_path: Path):
    """Load CSV samples from wav_to_csv output."""
    df = pd.read_csv(csv_path)
    assert "Sample" in df.columns
    return df["Sample"].to_numpy(dtype=np.float32)


@pytest.mark.parametrize("bit_depth", [16, 24, 32])
def test_wav_to_csv(tmp_path, bit_depth):
    wav_path = tmp_path / f"test_{bit_depth}.wav"
    csv_path = tmp_path / f"test_{bit_depth}.csv"

    expected, bd = write_test_wav(wav_path, bit_depth)

    result = subprocess.run(
        ["./cpp/audio/wav_to_csv", str(wav_path), str(csv_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"wav_to_csv failed: {result.stderr}"

    got = load_csv(csv_path)
    assert len(got) == len(expected)

    for i, (w, c) in enumerate(zip(expected, got)):
        if np.isclose(w, 0.0):
            assert np.isclose(c, 0.0), f"Mismatch at {i}: WAV={w}, CSV={c}"
        else:
            assert not np.isclose(c, 0.0), f"Unexpected zero at {i}: WAV={w}, CSV={c}"

