#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative AI Demo for Crescent Moon Images
Author: Julia Wen
Date: 2025-09-03
Description:
    Fully optimized synthetic crescent/no-crescent image generator using Stable Diffusion.
    Designed for stability, speed, and automation across CPU, CUDA, and Apple MPS.

Notes on Safety Checker:
    - Currently, this script does NOT enable the Hugging Face safety checker.
    - Warning message may appear:
        "You have disabled the safety checker..."
    - This is intentional for local/controlled experimentation.
    - If sharing results publicly or in a service, users MUST enable
      a safety checker or filter images according to the Stable Diffusion license.
    - Reference: https://github.com/huggingface/diffusers/pull/254
"""

import os
import torch
import random
import csv
import argparse
import signal
from tqdm import tqdm
from diffusers import StableDiffusionPipeline  # safety checker intentionally omitted

torch.set_num_threads(1)  # safer on macOS

stop_requested = False
def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True
    print("\n⚠️  Ctrl-C detected. Exiting gracefully after current image...")

signal.signal(signal.SIGINT, signal_handler)

# -------------------- Command-line arguments --------------------
parser = argparse.ArgumentParser(description="Synthetic Crescent/No-Crescent Image Generator")
parser.add_argument("--output_dir", type=str, default="dataset_gen", help="Directory to save generated images")
parser.add_argument("--images_per_prompt", type=int, default=3, help="Number of images per prompt")
parser.add_argument("--steps", type=int, default=None, help="Number of inference steps (auto if not set)")
parser.add_argument("--height", type=int, default=None, help="Image height (auto if not set)")
parser.add_argument("--width", type=int, default=None, help="Image width (auto if not set)")
parser.add_argument("--force_model", type=str, default=None, help="Force model name (overrides auto selection)")
parser.add_argument("--crescent_only", action="store_true", help="Generate only crescent images")
parser.add_argument("--user_prompt", type=str, default=None, help="Custom prompt for image generation")
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
IMAGES_PER_PROMPT = args.images_per_prompt

# -------------------- Device detection --------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# -------------------- Auto model & settings --------------------
if args.force_model:
    MODEL_NAME = args.force_model
elif device == "cpu":
    MODEL_NAME = "stabilityai/sd-turbo"
else:
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"

if device == "cpu":
    default_steps, default_height, default_width = 12, 384, 384
elif device == "mps":
    default_steps, default_height, default_width = 20, 512, 512
else:  # CUDA
    default_steps, default_height, default_width = 50, 512, 512

num_inference_steps = args.steps or default_steps
height = args.height or default_height
width = args.width or default_width

print(f"Model → {MODEL_NAME}")
print(f"Settings → steps={num_inference_steps}, size={height}x{width}")

# -------------------- Prepare folders --------------------
os.makedirs(os.path.join(OUTPUT_DIR, "crescent"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "no_crescent"), exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "generation_log.csv")
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "prompt_idx", "img_num", "seed", "filename"])

# -------------------- Load pipeline --------------------
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME)
# Move to device and set precision without triggering warnings
pipe = pipe.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
pipe.enable_attention_slicing()

# -------------------- Prompts --------------------
prompts = {
    "crescent": [
        "A night sky with a small crescent moon, realistic, clear",
        "A night sky with a glowing crescent moon over mountains"
    ],
    "no_crescent": [
        "A night sky without any moon, realistic",
        "A starry night sky with clouds, no moon"
    ]
}

# -------------------- Generation --------------------
def generate_images(prompt, label, prompt_idx, global_pbar):
    global stop_requested
    for img_num in range(IMAGES_PER_PROMPT):
        if stop_requested:
            break

        seed = random.randint(0, 2**32 - 1)
        filename = f"{label}_{prompt_idx+1}_{img_num+1}_{seed}.png"
        output_path = os.path.join(OUTPUT_DIR, label, filename)

        if os.path.exists(output_path):
            global_pbar.update(1)
            continue

        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"\n→ Generating {label} (prompt {prompt_idx+1}, img {img_num+1}, seed {seed})")

        # -------------------- Internal negative prompt --------------------
        # Suppress full moons for crescent generation
        if label == "crescent":
            negative = "full moon, round moon, circle, sphere"
        else:
            negative = None

        image = pipe(
            prompt,
            negative_prompt=negative,
            guidance_scale=7.5,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width
        ).images[0]

        image.save(output_path)
        print(f"   Saved → {output_path}")

        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label, prompt_idx, img_num, seed, filename])

        if device == "cuda":
            torch.cuda.empty_cache()

        global_pbar.update(1)

# -------------------- Run --------------------
active_prompts = {}
if args.crescent_only:
    active_prompts["crescent"] = prompts["crescent"]
else:
    active_prompts = prompts

if args.user_prompt:
    active_prompts = {"crescent": [args.user_prompt]}

total_images = sum(len(p) for p in active_prompts.values()) * IMAGES_PER_PROMPT
with tqdm(total=total_images, desc="Global Progress", unit="img") as global_pbar:
    for label, label_prompts in active_prompts.items():
        for idx, prompt in enumerate(label_prompts):
            generate_images(prompt, label, idx, global_pbar)

print("\n✅ Synthetic dataset generation completed.")
print(f"📒 Log saved to {LOG_FILE}")

