#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: detect_crescent_classical.py
Author: Julia Wen
Date: 2025-08-28
Description: classical version of Crescent Detection (Classical HOG + SVM)
"""

import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Dataset paths
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = (128, 128)  # Resize all images

# Load images and labels
def load_images_and_labels(dataset_dir, classes, img_size):
    X, y, paths = [], [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            img = imread(fpath)
            if img.shape[-1] == 4:  # RGBA
                img = img[..., :3]
            img = resize(img, img_size)
            img_gray = rgb2gray(img)
            X.append(img_gray)
            y.append(label)
            paths.append(fpath)
    return np.array(X), np.array(y), paths

# Extract HOG features
def extract_hog_features(X):
    features = []
    for img in X:
        feat = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    return np.array(features)

# Main
X, y, image_paths = load_images_and_labels(DATASET_DIR, CLASSES, IMG_SIZE)
if len(X) == 0:
    print("No images found in dataset. Check your dataset directory.")
    exit()

X_feat = extract_hog_features(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

# Train linear SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_scaled, y)

# Predict and print results
y_pred = svm.predict(X_scaled)
y_prob = svm.predict_proba(X_scaled)

print("\nPredictions:")
for path, pred, prob in zip(image_paths, y_pred, y_prob):
    print(f"{os.path.basename(path)} -> {CLASSES[pred]} (confidence: {np.max(prob):.2f})")

