#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: video_depth_midas_mp4.py
Author: Julia Wen
Date: 2025-08-22
Description: 
Create a depth-visualized video from an input video using MiDaS.

Usage:
  python3 video_depth_midas_mp4.py input.mp4 output.mp4 [--model DPT_Large|DPT_Hybrid|MiDaS_small] [--debug]

Notes:
- Uses OpenCV for reading frames, MiDaS for depth, and imageio-ffmpeg for MP4 writing.
- Produces a valid MP4 (H.264) if ffmpeg is installed.
"""

import os
import sys
import math
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import imageio

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="input video path")
    p.add_argument("output", help="output video path (.mp4)")
    p.add_argument("--model", default="DPT_Large",
                   help="MiDaS model: DPT_Large, DPT_Hybrid, MiDaS_small")
    p.add_argument("--debug", action="store_true", help="print extra logs")
    return p.parse_args()

def load_midas(model_name: str, device: torch.device, debug=False):
    if debug:
        print(f"[midas] loading model '{model_name}' on {device} ...")
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if "DPT" in model_name else transforms.small_transform
    if debug:
        print("[midas] model and transforms ready")
    return midas, transform

def normalize_to_u8(depth: np.ndarray) -> np.ndarray:
    dmin, dmax = float(np.min(depth)), float(np.max(depth))
    if math.isclose(dmax, dmin):
        return np.zeros_like(depth, dtype=np.uint8)
    norm = (depth - dmin) / (dmax - dmin + 1e-8)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)

def run(inp: Path, outp: Path, model_name="DPT_Large", debug=False):
    if debug:
        print(f"[cwd] {Path.cwd().resolve()}")
        print(f"[in]  {inp.resolve() if inp.exists() else inp}")
        print(f"[out] {outp.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas, transform = load_midas(model_name, device, debug=debug)

    cap = cv.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {inp}")

    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps <= 1e-3 or np.isnan(fps):
        fps = 30.0
        if debug:
            print("[cap] invalid FPS reported; defaulting to 30.0")

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Input video reports zero size; cannot proceed.")

    writer = imageio.get_writer(str(outp), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    if debug:
        print(f"[writer] opened with libx264 at {fps:.3f} fps, size=({w},{h})")

    frames_written = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            batch = transform(img).to(device)

            pred = midas(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze(1).squeeze(0)

            depth = pred.detach().cpu().numpy()
            depth_u8 = normalize_to_u8(depth)
            depth_color = cv.applyColorMap(depth_u8, cv.COLORMAP_MAGMA)  # BGR

            # imageio expects RGB
            writer.append_data(cv.cvtColor(depth_color, cv.COLOR_BGR2RGB))
            frames_written += 1

            if debug and frames_written % 10 == 0:
                print(f"[proc] frames written: {frames_written}")

    cap.release()
    writer.close()

    if frames_written == 0:
        raise RuntimeError("Finished but no frames were written!")

    print(f"✅ Wrote {frames_written} frames to: {outp.resolve()}")

def main():
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    run(inp, outp, model_name=args.model, debug=args.debug)

if __name__ == "__main__":
    main()
