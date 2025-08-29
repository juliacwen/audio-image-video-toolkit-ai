
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: predict_crescent.py
Author: Julia Wen
Date: 2025-08-28
Description: 
    crescent moon detection with deep learning or neural network features
"""

import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
from skimage.util import random_noise
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import argparse

# Dataset configuration
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = (128, 128)

def load_images_and_labels(dataset_dir, classes, img_size, augment=False):
    X, y, image_paths, original_indices = [], [], [], []

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(class_dir, fname)
                img = imread(img_path)

                if img.shape[-1] == 4:
                    img = img[:, :, :3]

                imgs_to_use = [img]
                if augment:
                    imgs_to_use.append(np.fliplr(img))
                    for angle in [-15, 15]:
                        imgs_to_use.append(rotate(img, angle, resize=False))
                    imgs_to_use.append(random_noise(img, mode='s&p', amount=0.01))

                for im in imgs_to_use:
                    img_gray = rgb2gray(im)
                    img_resized = resize(img_gray, img_size, anti_aliasing=True)
                    features = hog(img_resized, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), orientations=9)
                    X.append(features)
                    y.append(label_idx)
                    image_paths.append(img_path)
                    if im is img:
                        original_indices.append(len(X)-1)

    return np.array(X), np.array(y), image_paths, original_indices

# Argument parser
parser = argparse.ArgumentParser(description="Predict crescent moon in image.")
parser.add_argument("--image", type=str, help="Path to an image to predict outside training set")
parser.add_argument(
    "--no-display",
    dest='no_display',
    action='store_true',
    help=(
        "Disable plotting images during training prediction"
    )
)
args = parser.parse_args()

# Load dataset
print("Loading images from dataset with augmentation...")
X, y, image_paths, original_indices = load_images_and_labels(DATASET_DIR, CLASSES, IMG_SIZE, augment=True)
print(f"Loaded {len(X)} samples (including augmented): {dict(zip(CLASSES, [np.sum(y==i) for i in 
range(len(CLASSES))]))}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_scaled, y)

# Predict on original images
X_original_scaled = X_scaled[original_indices]
image_paths_original = [image_paths[i] for i in original_indices]

y_pred = svm.predict(X_original_scaled)
y_prob = svm.predict_proba(X_original_scaled)

print("\nPrediction summary (training images):")
for idx, path in enumerate(image_paths_original):
    label_name = CLASSES[y_pred[idx]]
    prob = y_prob[idx][y_pred[idx]]
    print(f"{path} → {label_name}, probability={prob:.2f}")
    if not args.no_display:
        img = imread(path)
        plt.imshow(img)
        plt.title(f"{os.path.basename(path)}: {label_name} ({prob:.2f})")
        plt.axis('off')
        plt.show()

# Predict new image if provided
if args.image:
    new_img_path = args.image
    if not os.path.exists(new_img_path):
        print(f"Error: {new_img_path} does not exist")
    else:
        img = imread(new_img_path)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img_gray = rgb2gray(img)
        img_resized = resize(img_gray, IMG_SIZE, anti_aliasing=True)
        features = hog(img_resized, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), orientations=9)
        features_scaled = scaler.transform([features])
        y_pred_new = svm.predict(features_scaled)
        y_prob_new = svm.predict_proba(features_scaled)
        label_name = CLASSES[y_pred_new[0]]
        prob = y_prob_new[0][y_pred_new[0]]
        print(f"\n{new_img_path} → {label_name}, probability={prob:.2f}")
        if not args.no_display:
            plt.imshow(img)
            plt.title(f"{os.path.basename(new_img_path)}: {label_name} ({prob:.2f})")
            plt.axis('off')
            plt.show()

# Save model
joblib.dump(svm, "crescent_moon_model.pkl")
joblib.dump(scaler, "crescent_moon_scaler.pkl")
print("\nModel and scaler saved as 'crescent_moon_model.pkl' and 'crescent_moon_scaler.pkl'.")

