#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_crescent_pytorch_svm.py

HOG -> SVM (sklearn) wrapped in PyTorch-friendly script:
 - Uses original augmentation only for training
 - Computes HOG features (grayscale, resized 128x128, same params)
 - StandardScaler applied to features
 - Linear SVM with probability=True (replicates original SVM probabilities)
 - Predicts on ORIGINAL images only
 - Displays color images
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
from skimage.util import random_noise
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import argparse

# -------------------
# Config
# -------------------
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = (128, 128)
HOG_PARAMS = dict(pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9)
SCALER_PATH = "crescent_moon_scaler.pkl"
MODEL_PATH = "crescent_moon_model.pkl"

# -------------------
# Dataset Loader
# -------------------
def augment_image_list(img_color):
    imgs = [img_color]
    imgs.append(np.fliplr(img_color))
    for angle in (-15, 15):
        imgs.append(rotate(img_color, angle, resize=False, mode='edge'))
    imgs.append(random_noise(img_color, mode='s&p', amount=0.01))
    return imgs

def load_images_and_hog(dataset_dir, classes, img_size, augment=True):
    X_feats = []
    y = []
    image_paths = []
    original_indices = []

    for label_idx, cls in enumerate(classes):
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing folder: {class_dir}")
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            path = os.path.join(class_dir, fname)
            img = imread(path)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]

            imgs_to_use = augment_image_list(img) if augment else [img]
            for i_aug, im_color in enumerate(imgs_to_use):
                imf = im_color.astype(np.float32)
                if imf.max() > 1.0:
                    imf /= 255.0
                im_gray = rgb2gray(imf)
                im_gray_resized = resize(im_gray, img_size, anti_aliasing=True)
                feats = hog(im_gray_resized, **HOG_PARAMS)
                X_feats.append(feats.astype(np.float32))
                y.append(label_idx)
                image_paths.append(path)
                if i_aug == 0:
                    original_indices.append(len(X_feats)-1)

    return np.array(X_feats), np.array(y), image_paths, original_indices

# -------------------
# Main
# -------------------
def main(args):
    print("Loading images and extracting HOG (with augmentation)...")
    X, y, image_paths, original_indices = load_images_and_hog(DATASET_DIR, CLASSES, IMG_SIZE, augment=True)
    print(f"Total samples (including augmented): {len(X)}")
    print("Class counts (augmented):", dict(zip(CLASSES, [np.sum(y==i) for i in range(len(CLASSES))])))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Train linear SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_scaled, y)
    joblib.dump(svm, MODEL_PATH)
    print(f"SVM model saved to {MODEL_PATH}")

    # Predict on original images only
    X_orig_scaled = X_scaled[original_indices]
    image_paths_orig = [image_paths[i] for i in original_indices]
    y_prob = svm.predict_proba(X_orig_scaled)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nPrediction summary (training originals):")
    for idx, path in enumerate(image_paths_orig):
        pred_idx = y_pred[idx]
        label_name = CLASSES[pred_idx]
        prob = y_prob[idx][pred_idx]
        print(f"{path} → {label_name}, probability={prob:.2f}")
        if not args.no_display:
            img = imread(path)
            plt.imshow(img)
            plt.title(f"{os.path.basename(path)}: {label_name} ({prob:.2f})")
            plt.axis("off")
            plt.show()

    # Optional: predict new image
    if args.image:
        new_path = args.image
        if not os.path.exists(new_path):
            print(f"Error: {new_path} not found")
            return
        img = imread(new_path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        im_gray = rgb2gray(resize(img, IMG_SIZE, anti_aliasing=True))
        feats = hog(im_gray, **HOG_PARAMS).astype(np.float32)
        feats_scaled = scaler.transform([feats])
        prob = svm.predict_proba(feats_scaled)[0]
        pred_idx = np.argmax(prob)
        label_name = CLASSES[pred_idx]
        print(f"\n{new_path} → {label_name}, probability={prob[pred_idx]:.2f}")
        if not args.no_display:
            plt.imshow(img)
            plt.title(f"{os.path.basename(new_path)}: {label_name} ({prob[pred_idx]:.2f})")
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOG -> SVM (faithful to original)")
    parser.add_argument("--image", type=str, help="Optional new image")
    parser.add_argument("--no-display", action="store_true", help="Disable image display")
    args = parser.parse_args()
    main(args)

