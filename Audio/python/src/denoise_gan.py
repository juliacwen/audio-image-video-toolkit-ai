#!/usr/bin/env python3
"""
denoise_gan.py — Conv2D Denoising GAN
Author: Julia Wen
Date: 2025-09-23

Description:
    This script implements a convolutional 2D Generative Adversarial Network (GAN)
    for audio denoising. It operates on spectrograms of audio signals and supports
    both training and inference. Noisy audio can be synthetically generated from
    clean audio using Gaussian noise.
    Configurable parameters (dataset paths, model settings, training options, etc.)
    are provided via a YAML configuration file, with some values still hard-coded.

Features:
    - Dataset loader for noisy/clean audio pairs, computes log-magnitude spectrograms
    - Conv2D U-Net style generator for predicting noise in spectrograms
    - PatchGAN discriminator for adversarial training
    - Training loop with L1 reconstruction loss and adversarial loss
    - Inference with overlap-add strategy for long audio files
    - CLI support for dataset checking, training, and denoising inference

Dependencies:
    - Python 3.8+
    - PyTorch
    - NumPy
    - PyDub
    - pyyaml

CLI Examples:
    # Check dataset
    python3 denoise_gan.py --config config_denoise_gan.yaml --check_dataset

    # Train the GAN
    python3 denoise_gan.py --config config_denoise_gan.yaml --train

    # Denoise a WAV file
    python3 denoise_gan.py --config config_denoise_gan.yaml --infer \
        --noisy_wav data/noisy/clean_0_noisy.wav \
        --out_wav clean_0_noisy_denoised.wav
"""

import os
import sys
import yaml
import math
import numpy as np
from pydub import AudioSegment
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Helpers: Audio I/O
# ---------------------------
def load_audio_mono(path, target_sr):
    a = AudioSegment.from_file(path)
    a = a.set_frame_rate(target_sr).set_channels(1)
    samples = np.array(a.get_array_of_samples()).astype(np.float32)
    denom = float(2 ** (8 * a.sample_width - 1))
    return (samples / denom), target_sr

def save_wav(samples, path, sr):
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767.0).astype(np.int16)
    seg = AudioSegment(int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    seg.export(path, format="wav")

# ---------------------------
# Synthetic noise generation
# ---------------------------
def generate_noisy_from_clean(clean_path, noisy_path, noise_level, target_sr):
    """Generate a noisy WAV from clean audio by adding Gaussian noise."""
    samples, sr = load_audio_mono(clean_path, target_sr)
    rms = math.sqrt(float(np.mean(samples ** 2)) + 1e-12)
    noise = np.random.randn(len(samples)).astype(np.float32) * (noise_level * (rms + 1e-12))
    noisy = np.clip(samples + noise, -1.0, 1.0)
    if not noisy_path.lower().endswith(".wav"):
        noisy_path = os.path.splitext(noisy_path)[0] + ".wav"
    save_wav(noisy, noisy_path, sr)
    return noisy_path

# ---------------------------
# Dataset: noisy/clean pairs with STFT
# ---------------------------
class SpectrogramDataset(Dataset):
    """Dataset: load clean + noisy WAVs, compute log-magnitude spectrograms."""
    def __init__(self, clean_dir, noisy_dir, sample_rate, n_fft, hop_length,
                 fixed_frames=None, duration=None, noise_level=0.05, spec_power=1.0):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sr = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop = int(hop_length)
        self.spec_power = float(spec_power)
        self.noise_level = float(noise_level)

        if fixed_frames is not None:
            self.fixed_frames = int(fixed_frames)
            self.fixed_samples = self.n_fft + self.hop * (self.fixed_frames - 1)
        elif duration is not None:
            self.fixed_samples = int(round(float(duration) * self.sr))
            self.fixed_frames = 1 + max(0, (self.fixed_samples - self.n_fft) // self.hop)
        else:
            raise RuntimeError("Either fixed_frames or duration must be provided")

        VALID = (".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac")
        clean_files = [f for f in sorted(os.listdir(clean_dir)) if f.lower().endswith(VALID)]
        if len(clean_files) == 0:
            raise RuntimeError(f"No clean audio files found in {clean_dir}")

        # create noisy files if missing
        os.makedirs(noisy_dir, exist_ok=True)
        self.pairs = []
        for cf in clean_files:
            base = os.path.splitext(cf)[0]
            noisy_name = base + "_noisy.wav"
            noisy_path = os.path.join(noisy_dir, noisy_name)
            clean_path = os.path.join(clean_dir, cf)
            if not os.path.exists(noisy_path):
                generate_noisy_from_clean(clean_path, noisy_path, self.noise_level, self.sr)
            self.pairs.append((noisy_path, clean_path))

    def __len__(self):
        return len(self.pairs)

    def _pad_or_truncate(self, arr):
        if len(arr) < self.fixed_samples:
            pad = self.fixed_samples - len(arr)
            return np.pad(arr, (0, pad))
        return arr[:self.fixed_samples]

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy_wav, _ = load_audio_mono(noisy_path, self.sr)
        clean_wav, _ = load_audio_mono(clean_path, self.sr)
        noisy_wav = self._pad_or_truncate(noisy_wav)
        clean_wav = self._pad_or_truncate(clean_wav)

        # STFT → magnitude → power → log
        window = torch.hann_window(self.n_fft)
        noisy_t = torch.from_numpy(noisy_wav).float()
        clean_t = torch.from_numpy(clean_wav).float()

        noisy_stft = torch.stft(noisy_t, n_fft=self.n_fft, hop_length=self.hop, return_complex=True, window=window)
        clean_stft = torch.stft(clean_t, n_fft=self.n_fft, hop_length=self.hop, return_complex=True, window=window)

        mag_noisy = torch.abs(noisy_stft)
        mag_clean = torch.abs(clean_stft)

        if self.spec_power != 1.0:
            mag_noisy = mag_noisy.pow(self.spec_power)
            mag_clean = mag_clean.pow(self.spec_power)

        log_noisy = torch.log1p(mag_noisy).unsqueeze(0)  # (1, F, T)
        log_clean = torch.log1p(mag_clean).unsqueeze(0)

        return log_noisy, log_clean

# ---------------------------
# Generator: Conv2D U-Net style
# ---------------------------
class Conv2dGenerator(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        # simple encoder-decoder U-Net style
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Conv2d(base_ch, in_ch, 3, padding=1)  # predict noise

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        out = self.dec2(d1)
        return out

# ---------------------------
# Discriminator: PatchGAN Conv2D
# ---------------------------
class Conv2dDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch*2, 1, 4, padding=1)  # output single-channel patch map
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Training loop
# ---------------------------
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device","cuda")).startswith("cuda") else "cpu")
    ds_cfg = cfg["dataset"]

    dataset = SpectrogramDataset(
        clean_dir=ds_cfg["clean_dir"],
        noisy_dir=ds_cfg.get("noisy_dir","data/noisy"),
        sample_rate=ds_cfg["sample_rate"],
        n_fft=cfg.get("n_fft",1024),
        hop_length=cfg.get("hop_length",512),
        fixed_frames=cfg.get("fixed_frames",256),
        duration=ds_cfg.get("duration",1.0),
        noise_level=ds_cfg.get("noise_level",0.05),
        spec_power=cfg.get("spec_power",2)
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    G = Conv2dGenerator(base_ch=cfg["gen_features"]).to(device)
    D = Conv2dDiscriminator(base_ch=cfg["disc_features"]).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate"], betas=(0.5,0.999))
    optD = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate"], betas=(0.5,0.999))
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    print(f"Device: {device}, dataset size: {len(dataset)}")
    for ep in range(cfg["epochs"]):
        for i, (noisy_log, clean_log) in enumerate(loader):
            noisy_log = noisy_log.to(device)
            clean_log = clean_log.to(device)

            # ----------------------
            # Train discriminator
            # ----------------------
            with torch.no_grad():
                fake = noisy_log - G(noisy_log)
            real_pred = D(clean_log)
            fake_pred = D(fake)
            D_loss = (bce(real_pred, torch.ones_like(real_pred)) +
                      bce(fake_pred, torch.zeros_like(fake_pred))) * 0.5

            optD.zero_grad()
            D_loss.backward()
            optD.step()

            # ----------------------
            # Train generator
            # ----------------------
            pred_noise = G(noisy_log)
            clean_pred = noisy_log - pred_noise
            adv_loss = bce(D(clean_pred), torch.ones_like(D(clean_pred)))
            rec_loss = l1(clean_pred, clean_log)
            G_loss = cfg.get("adv_weight",1.0)*adv_loss + cfg.get("rec_weight",100.0)*rec_loss

            optG.zero_grad()
            G_loss.backward()
            optG.step()

            if i % 10 == 0:
                print(f"[{ep+1}/{cfg['epochs']}] Step {i} | D_loss={D_loss.item():.6f} G_loss={G_loss.item():.6f}")

    # save generator
    out_path = cfg.get("model_out","generator_denoise.pth")
    torch.save(G.state_dict(), out_path)
    print(f"✅ Generator saved: {out_path}")

# ---------------------------
# Inference
# ---------------------------
def infer(cfg, model_path, noisy_wav, out_wav, infer_block_sec=None, spectral_floor=0.01):
    import torch
    import numpy as np

    device = torch.device(
        "cuda" if torch.cuda.is_available() and str(cfg.get("device","cuda")).startswith("cuda") else "cpu"
    )
    ds_cfg = cfg["dataset"]
    sr = int(ds_cfg["sample_rate"])

    # Determine block size
    block_sec = float(infer_block_sec or ds_cfg.get("duration",1.0))
    fixed_samples = int(sr * block_sec)

    # Load model
    G = Conv2dGenerator(base_ch=cfg["gen_features"])
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.to(device).eval()

    # Load audio
    samples, _ = load_audio_mono(noisy_wav, sr)
    audio_len = len(samples)

    # If clip is shorter than block size, process as a single block
    if audio_len <= fixed_samples:
        x = torch.from_numpy(samples).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            clean_pred = x - G(x)
        out = clean_pred.squeeze().cpu().numpy()
        out = np.sign(out) * np.maximum(np.abs(out), spectral_floor)
        save_wav(out, out_wav, sr)
        print(f"✅ Denoised short audio saved: {out_wav} | Block duration: {block_sec}s | spectral floor={spectral_floor}")
        return

    # For longer audio, use overlap-add
    step = fixed_samples // 2  # 50% overlap
    out_buffer = np.zeros(audio_len + step)
    weight = np.zeros_like(out_buffer)

    idx = 0
    while idx < audio_len:
        block = samples[idx: idx + fixed_samples]
        orig_len = len(block)

        # Pad only if necessary
        if len(block) < fixed_samples:
            block = np.pad(block, (0, fixed_samples - len(block)))

        x = torch.from_numpy(block).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_noise = G(x)

        clean_block = block - pred_noise.squeeze().cpu().numpy()[:len(block)]
        clean_block = np.sign(clean_block) * np.maximum(np.abs(clean_block), spectral_floor)

        # RMS normalization
        rms_in = block.std() + 1e-8
        rms_out = clean_block.std() + 1e-8
        if rms_out > 1e-6:
            clean_block = clean_block * (rms_in / rms_out)

        end_idx = min(idx + fixed_samples, len(out_buffer))
        length_to_add = end_idx - idx
        out_buffer[idx:end_idx] += clean_block[:length_to_add]
        weight[idx:end_idx] += 1.0

        idx += step

    # Normalize by overlap weights
    out = out_buffer[:audio_len] / np.maximum(weight[:audio_len], 1e-8)
    save_wav(out, out_wav, sr)
    print(f"✅ Denoised audio saved: {out_wav} | Block duration: {block_sec}s | 50% overlap | spectral floor={spectral_floor}")

# ---------------------------
# Dataset check
# ---------------------------
def check_dataset(cfg):
    ds_cfg = cfg["dataset"]
    dataset = SpectrogramDataset(
        clean_dir=ds_cfg["clean_dir"],
        noisy_dir=ds_cfg.get("noisy_dir","data/noisy"),
        sample_rate=ds_cfg["sample_rate"],
        n_fft=cfg.get("n_fft",1024),
        hop_length=cfg.get("hop_length",512),
        fixed_frames=cfg.get("fixed_frames",256),
        duration=ds_cfg.get("duration",1.0),
        noise_level=ds_cfg.get("noise_level",0.05),
        spec_power=cfg.get("spec_power",2)
    )
    print(f"✅ Dataset OK. Samples: {len(dataset)}, Fixed frames: {dataset.fixed_frames}, fixed_samples: {dataset.fixed_samples}")
    for i, (nlog, clog) in enumerate(dataset):
        print(f"Example {i}: noisy_log shape={nlog.shape}, clean_log shape={clog.shape}")
        if i >= 5:
            break

# ---------------------------
# CLI
# ---------------------------
def load_config(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--check_dataset", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--noisy_wav", type=str)
    parser.add_argument("--out_wav", type=str)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--infer_block_sec", type=float, default=None,
                    help="Block duration in seconds for inference (short files benefit from longer blocks)")

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.check_dataset:
        check_dataset(cfg)
        sys.exit(0)
    if args.train:
        train(cfg)
        sys.exit(0)
    if args.infer:
        if not args.noisy_wav or not args.out_wav:
            print("Provide --noisy_wav and --out_wav for inference")
            sys.exit(1)
        model_path = args.model or cfg.get("model_out","generator_denoise.pth")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            sys.exit(1)
        infer(cfg, model_path, args.noisy_wav, args.out_wav, infer_block_sec=args.infer_block_sec)
        sys.exit(0)

    parser.print_help()

