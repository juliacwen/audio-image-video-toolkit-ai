#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_crescent_tf.py
Author: Julia Wen
Date: 2025-09-02
Description:
TensorFlow classic classifier HOG with data augmentation.

Usage:
    python predict_crescent_tf.py            # train and show training predictions
    python predict_crescent_tf.py --no-display
    python predict_crescent_tf.py --image /path/to/new.jpg
"""

import os
import sys
import argparse
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
from skimage.util import random_noise
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# ---- Config ----
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]   # MUST match folder names and desired label order
IMG_SIZE = (128, 128)                  # height, width for HOG resize
HOG_PARAMS = dict(pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9)
MODEL_PATH = "crescent_moon_model_svc.pkl"
SCALER_PATH = "crescent_moon_scaler.pkl"
RNG_SEED = 42

# reproducibility
np.random.seed(RNG_SEED)

# ---- Helpers ----
def augment_color_image_list(img_color):
    """
    Return list: [original, flipped, rotated -15, rotated +15, salt-and-pepper]
    Each element is a color image (float or uint). We keep behavior similar to original script:
    augmentation applied on the color image, then later converted to grayscale+resized for HOG.
    """
    imgs = [img_color]
    try:
        imgs.append(np.fliplr(img_color))
    except Exception:
        imgs.append(img_color.copy())
    for angle in (-15, 15):
        imgs.append(rotate(img_color, angle, resize=False, mode='edge'))
    imgs.append(random_noise(img_color, mode='s&p', amount=0.01))
    return imgs

def load_images_and_hog(dataset_dir, classes, img_size, augment=True):
    """
    Walk class directories and return:
      X_feats: ndarray (n_samples, n_features) HOG features (augmented samples included)
      y: ndarray (n_samples,)
      image_paths: list[str] of length n_samples (original path repeated for augmented samples)
      original_indices: list[int] indices into X_feats that correspond to the original file (one per original file)
    """
    X_feats = []
    y = []
    image_paths = []
    original_indices = []

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected dataset directory: {class_dir}")
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            path = os.path.join(class_dir, fname)
            img = imread(path)
            # Ensure color image (H,W,3)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]

            # Build augmented list (first entry is the original)
            imgs_to_use = augment_color_image_list(img) if augment else [img]

            for i_aug, im_color in enumerate(imgs_to_use):
                # prepare grayscale resized image for HOG
                # skimage.rgb2gray expects floats in [0,1] or uints; ensure float in [0,1]
                im_float = im_color.astype(np.float32)
                if im_float.max() > 1.0:
                    im_float = im_float / 255.0
                im_gray = rgb2gray(im_float)  # float in [0,1]
                im_gray_resized = resize(im_gray, img_size, anti_aliasing=True)  # float in [0,1]
                feats = hog(im_gray_resized, **HOG_PARAMS)
                X_feats.append(feats.astype(np.float32))
                y.append(label_idx)
                image_paths.append(path)
                # mark the index of the original (first augmented entry)
                if i_aug == 0:
                    original_indices.append(len(X_feats) - 1)

    X_feats = np.asarray(X_feats, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X_feats, y, image_paths, original_indices

def display_color_image(path, title=None):
    img = imread(path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    disp = resize(img, IMG_SIZE, anti_aliasing=True, preserve_range=True).astype(np.float32)
    if disp.max() > 1.0:
        disp = disp / 255.0
    plt.imshow(disp)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# ---- Main ----
def main(args):
    print("Loading images and extracting HOG features (with augmentation)...")
    X_feats, y, image_paths, original_indices = load_images_and_hog(DATASET_DIR, CLASSES, IMG_SIZE, augment=True)

    if X_feats.size == 0:
        raise RuntimeError("No images found. Ensure dataset/<class>/ images exist.")

    print(f"Total samples (including augmented): {len(X_feats)}")
    unique, counts = np.unique(y, return_counts=True)
    print("Class counts (augmented):", dict(zip(unique.tolist(), counts.tolist())))

    # Diagnostics for image -> HOG pipeline
    print("HOG diagnostics: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
        float(X_feats.mean()), float(X_feats.std()), float(X_feats.min()), float(X_feats.max())
    ))
    if np.allclose(X_feats, 0):
        print("WARNING: HOG features are near zero — check image pipeline.", file=sys.stderr)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feats)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")

    # Train SVM with probabilities (Platt scaling)
    print("Training SVM (linear kernel, probability=True)...")
    svm = SVC(kernel='linear', probability=True, random_state=RNG_SEED)
    svm.fit(X_scaled, y)
    joblib.dump(svm, MODEL_PATH)
    print(f"Saved SVM model to {MODEL_PATH}")

    # Predict on originals only (one line per original file)
    X_orig_scaled = X_scaled[original_indices]
    orig_paths = [image_paths[i] for i in original_indices]

    y_pred = svm.predict(X_orig_scaled)
    y_prob = svm.predict_proba(X_orig_scaled)

    print("\nPrediction summary (training images):")
    for idx, path in enumerate(orig_paths):
        pred_idx = int(y_pred[idx])
        prob_pred = float(y_prob[idx][pred_idx])  # probability for the predicted class
        label_name = CLASSES[pred_idx]
        print(f"{path} → {label_name}, probability={prob_pred:.2f}")
        if not args.no_display:
            display_color_image(path, title=f"{os.path.basename(path)}: {label_name} ({prob_pred:.2f})")

    # Optional: predict a single new image
    if args.image:
        new_path = args.image
        if not os.path.exists(new_path):
            print(f"Error: {new_path} not found", file=sys.stderr)
        else:
            img = imread(new_path)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            im_float = img.astype(np.float32)
            if im_float.max() > 1.0:
                im_float /= 255.0
            im_gray = rgb2gray(im_float)
            im_gray_resized = resize(im_gray, IMG_SIZE, anti_aliasing=True)
            feats = hog(im_gray_resized, **HOG_PARAMS)
            feats_scaled = scaler.transform([feats])
            pred = svm.predict(feats_scaled)[0]
            prob = svm.predict_proba(feats_scaled)[0][pred]
            label = CLASSES[int(pred)]
            print(f"\n{new_path} → {label}, probability={float(prob):.2f}")
            if not args.no_display:
                display_color_image(new_path, title=f"{os.path.basename(new_path)}: {label} ({float(prob):.2f})")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed crescent detection (HOG -> SVM)")
    parser.add_argument("--image", type=str, default=None, help="Optional image to predict")
    parser.add_argument("--no-display", action="store_true", help="Disable image display")
    args = parser.parse_args()
    main(args)
