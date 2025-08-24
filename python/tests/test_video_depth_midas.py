import os
import subprocess
import cv2
import numpy as np

# Path to the script under python/video/
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "video", "video_depth_midas.py")

def make_dummy_video(path, frames=5, width=64, height=64):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 5.0, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (i*30 % 255, i*60 % 255, i*90 % 255)
        out.write(frame)
    out.release()

def test_video_depth_midas(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"

    make_dummy_video(str(input_video))

    # Run the video depth script via subprocess (same style as python/audio tests)
    subprocess.run([
        "python3",
        SCRIPT,
        str(input_video),
        str(output_video),
        "--model", "DPT_Hybrid",
        "--debug"
    ], check=True)

    assert output_video.exists()
    assert output_video.stat().st_size > 0
