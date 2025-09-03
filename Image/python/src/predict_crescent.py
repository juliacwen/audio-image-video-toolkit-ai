#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: predict_crescent.py
Author: Julia Wen
Date: 2025-08-28
Description:
    This script performs crescent moon detection using a Support Vector Machine (SVM) classifier.
    
    The process includes:
    - Loading and preprocessing images from a dataset directory.
    - Applying data augmentation (horizontal flipping, rotation, and noise addition).
    - Extracting Histogram of Oriented Gradients (HOG) features from the images.
    - Training an SVM classifier on the extracted features.
    - Predicting the class (crescent or no_crescent) for new input images or training samples.
    
    The classifier is trained using augmented data to improve generalization. Augmentation includes:
    - Flipping images horizontally.
    - Rotating images by ±15 degrees.
    - Adding salt-and-pepper noise to simulate variations in image quality.
    
    After training, the model can be used to predict the class of an image, displaying the result along with the predicted probability.
    
    The trained SVM model and scaler are saved as 'crescent_moon_model.pkl' and 'crescent_moon_scaler.pkl' for later use.

Usage:
    To predict an image outside the training set:
    python predict_crescent.py --image /path/to/image.jpg
    
    To disable image display during prediction:
    python predict_crescent.py --image /path/to/image.jpg --no-display
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

# -------------------------------
# Configuration
# -------------------------------
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = (128, 128)  # Image size for resizing (height, width)

# -------------------------------
# Functions
# -------------------------------
def load_images_and_labels(dataset_dir, classes, img_size, augment=False):
    """
    Load images from dataset, optionally apply augmentation, and extract HOG features.

    Parameters:
        dataset_dir (str): Path to the dataset directory.
        classes (list): List of class names corresponding to subfolders.
        img_size (tuple): Target size for resizing images.
        augment (bool): Whether to apply simple data augmentation.

    Returns:
        X (np.ndarray): Feature vectors (HOG features).
        y (np.ndarray): Corresponding class labels.
        image_paths (list): Original file paths for each sample.
        original_indices (list): Indices corresponding to non-augmented images.
    """
    X, y, image_paths, original_indices = [], [], [], []

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(class_dir, fname)
                img = imread(img_path)

                # Remove alpha channel if present
                if img.shape[-1] == 4:
                    img = img[:, :, :3]

                imgs_to_use = [img]

                if augment:
                    # Horizontal flip
                    imgs_to_use.append(np.fliplr(img))
                    # Rotations
                    for angle in [-15, 15]:
                        imgs_to_use.append(rotate(img, angle, resize=False))
                    # Salt-and-pepper noise
                    imgs_to_use.append(random_noise(img, mode='s&p', amount=0.01))

                for im in imgs_to_use:
                    # Convert to grayscale
                    img_gray = rgb2gray(im)
                    # Resize image
                    img_resized = resize(img_gray, img_size, anti_aliasing=True)
                    # Extract HOG features
                    features = hog(img_resized, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), orientations=9)
                    X.append(features)
                    y.append(label_idx)
                    image_paths.append(img_path)
                    # Track original (non-augmented) image indices
                    if im is img:
                        original_indices.append(len(X) - 1)

    return np.array(X), np.array(y), image_paths, original_indices

def predict_and_display(svm, scaler, img_path, classes, img_size, display=True):
    """
    Predict a single image and optionally display it with the predicted label.

    Parameters:
        svm (SVC): Trained SVM model.
        scaler (StandardScaler): Fitted scaler for feature normalization.
        img_path (str): Path to the image.
        classes (list): Class names.
        img_size (tuple): Target image size for HOG extraction.
        display (bool): Whether to display the image with matplotlib.

    Returns:
        label_name (str): Predicted class name.
        prob (float): Probability of predicted class.
    """
    img = imread(img_path)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img_gray = rgb2gray(img)
    img_resized = resize(img_gray, img_size, anti_aliasing=True)
    features = hog(img_resized, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), orientations=9)
    features_scaled = scaler.transform([features])

    y_pred = svm.predict(features_scaled)
    y_prob = svm.predict_proba(features_scaled)
    label_name = classes[y_pred[0]]
    prob = y_prob[0][y_pred[0]]

    if display:
        plt.imshow(img)
        plt.title(f"{os.path.basename(img_path)}: {label_name} ({prob:.2f})")
        plt.axis('off')
        plt.show()

    return label_name, prob

# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Predict crescent moon in image.")
    parser.add_argument("--image", type=str, help="Path to an image to predict outside training set")
    parser.add_argument(
        "--no-display",
        dest='no_display',
        action='store_true',
        help="Disable plotting images during training prediction"
    )
    args = parser.parse_args()

    # Load dataset with augmentation
    print("Loading images from dataset with augmentation...")
    X, y, image_paths, original_indices = load_images_and_labels(DATASET_DIR, CLASSES, IMG_SIZE, augment=True)
    print(f"Loaded {len(X)} samples (including augmented): "
          f"{dict(zip(CLASSES, [np.sum(y == i) for i in range(len(CLASSES))]))}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM classifier
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
            predict_and_display(svm, scaler, path, CLASSES, IMG_SIZE, display=True)

    # Predict a new image if provided
    if args.image:
        new_img_path = args.image
        if not os.path.exists(new_img_path):
            print(f"Error: {new_img_path} does not exist")
        else:
            label_name, prob = predict_and_display(svm, scaler, new_img_path, CLASSES, IMG_SIZE, display=not args.no_display)
            print(f"\n{new_img_path} → {label_name}, probability={prob:.2f}")

    # Save trained model and scaler
    joblib.dump(svm, "crescent_moon_model.pkl")
    joblib.dump(scaler, "crescent_moon_scaler.pkl")
    print("\nModel and scaler saved as 'crescent_moon_model.pkl' and 'crescent_moon_scaler.pkl'.")

