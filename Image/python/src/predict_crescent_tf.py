#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_crescent_tf.py
Author: Julia Wen
Date: 2025-09-07
Description:
TensorFlow CNN classifier with data augmentation.
Trains on color images directly.
Saves trained model to crescent_moon_cnn.h5.
Supports external image prediction with --image.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
from skimage.util import random_noise
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

# ----------------------------
# Config
# ----------------------------
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = (128, 128)
MODEL_PATH = "crescent_moon_cnn.h5"
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ----------------------------
# Helpers
# ----------------------------
def augment_color_image_list(img_color):
    imgs = [img_color]
    try:
        imgs.append(np.fliplr(img_color))
    except Exception:
        imgs.append(img_color.copy())
    for angle in (-15, 15):
        imgs.append(rotate(img_color, angle, resize=False, mode='edge'))
    imgs.append(random_noise(img_color, mode='s&p', amount=0.01))
    return imgs

def load_images(dataset_dir, classes, img_size, augment=True):
    X = []
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
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]

            imgs_to_use = augment_color_image_list(img) if augment else [img]

            for i_aug, im_color in enumerate(imgs_to_use):
                im_float = im_color.astype(np.float32)
                if im_float.max() > 1.0:
                    im_float /= 255.0
                im_resized = resize(im_float, img_size, anti_aliasing=True)
                X.append(im_resized)
                y.append(label_idx)
                image_paths.append(path)
                if i_aug == 0:
                    original_indices.append(len(X)-1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, image_paths, original_indices

def display_image(img_array, title=None):
    img_array = np.clip(img_array, 0.0, 1.0)
    plt.imshow(img_array)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# ----------------------------
# Build CNN
# ----------------------------
def build_cnn(input_shape=(128,128,3), num_classes=2):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# Main
# ----------------------------
def main(args):
    X, y, image_paths, original_indices = load_images(DATASET_DIR, CLASSES, IMG_SIZE, augment=True)
    if X.size == 0:
        raise RuntimeError("No images found in dataset.")

    print(f"Total samples (including augmented): {len(X)}")
    print(f"Number of original images: {len(original_indices)}")

    model = build_cnn(input_shape=X.shape[1:], num_classes=len(CLASSES))
    model.summary()

    # Train
    model.fit(X, y, batch_size=8, epochs=20, verbose=1)

    # Save model
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Prediction on original images
    print("\nPrediction summary (original training images):")
    X_orig = X[original_indices]
    for idx, x_img in enumerate(X_orig):
        x_input = np.expand_dims(x_img, axis=0)
        prob = model.predict(x_input, verbose=0)[0]
        pred_idx = int(np.argmax(prob))
        label = CLASSES[pred_idx]
        print(f"{image_paths[original_indices[idx]]} → {label}, probability={prob[pred_idx]:.2f}")
        if not args.no_display:
            display_image(x_img, title=f"{label} ({prob[pred_idx]:.2f})")

    # Optional prediction on external image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: {args.image} not found")
        else:
            img = imread(args.image)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0
            img = resize(img, IMG_SIZE, anti_aliasing=True)
            x_input = np.expand_dims(img, axis=0)
            prob = model.predict(x_input, verbose=0)[0]
            pred_idx = int(np.argmax(prob))
            label = CLASSES[pred_idx]
            print(f"\n{args.image} → {label}, probability={prob[pred_idx]:.2f}")
            if not args.no_display:
                display_image(img, title=f"{label} ({prob[pred_idx]:.2f})")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN crescent detection with data augmentation")
    parser.add_argument("--image", type=str, default=None, help="Optional external image to predict")
    parser.add_argument("--no-display", action="store_true", help="Disable display of images")
    args = parser.parse_args()
    main(args)

