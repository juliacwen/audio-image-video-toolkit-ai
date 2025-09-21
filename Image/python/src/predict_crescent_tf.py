#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_crescent_tf.py
Author: Julia Wen
Date: 2025-09-07
Description:
TensorFlow CNN classifier with data augmentation.
Trains on color images directly.
Saves trained model to crescent_moon_cnn.keras.
Supports external image prediction with --image.

Revision History:
2025-09-20 - Updates:
    1. Model saving format changed from .h5 to .keras
    2. Added results saving feature:
        - All predictions are saved in <script_name>_result.txt inside test_output/
        - Predicted images (original training + optional external images) saved as high-resolution PNG with '_tf' suffix, showing label and probability
    3. Added YAML configuration support:
        - All configurable parameters (dataset, image, CNN, training, augmentation, model path) now loaded from config_predict_crescent_tf.yaml
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.util import random_noise
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
import yaml

# ----------------------------
# Load Config from YAML
# ----------------------------
CONFIG_FILE = "config_predict_crescent_tf.yaml"
with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

# ----------------------------
# Dataset configuration
# ----------------------------
DATASET_DIR = cfg['dataset']['dir']
CLASSES = cfg['dataset']['classes']

# ----------------------------
# Image configuration
# ----------------------------
IMG_HEIGHT = cfg['image']['height']
IMG_WIDTH = cfg['image']['width']
IMG_CHANNELS = cfg['image']['channels']
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
MAX_PIXEL_VALUE = cfg['image']['max_pixel_value']

# ----------------------------
# CNN architecture parameters
# ----------------------------
FILTERS_LAYER1 = cfg['cnn']['filters']['layer1']
FILTERS_LAYER2 = cfg['cnn']['filters']['layer2']
FILTERS_LAYER3 = cfg['cnn']['filters']['layer3']
KERNEL_SIZE = tuple(cfg['cnn']['kernel_size'])
DENSE_UNITS = cfg['cnn']['dense_units']
DROPOUT_RATE = cfg['cnn']['dropout_rate']

# ----------------------------
# Training parameters
# ----------------------------
LEARNING_RATE = cfg['training']['learning_rate']
BATCH_SIZE = cfg['training']['batch_size']
EPOCHS = cfg['training']['epochs']
RNG_SEED = cfg['training']['rng_seed']
np.random.seed(RNG_SEED)

# ----------------------------
# Data augmentation parameters
# ----------------------------
ROTATION_ANGLES = tuple(cfg['augmentation']['rotation_angles'])
SNP_NOISE_AMOUNT = cfg['augmentation']['snp_noise_amount']

# ----------------------------
# Model parameters
# ----------------------------
MODEL_PATH = cfg['model']['path']

# ----------------------------
# Helpers
# ----------------------------
def augment_color_image_list(img_color):
    imgs = [img_color]
    try:
        imgs.append(np.fliplr(img_color))
    except Exception:
        imgs.append(img_color.copy())
    for angle in ROTATION_ANGLES:
        imgs.append(rotate(img_color, angle, resize=False, mode='edge'))
    imgs.append(random_noise(img_color, mode='s&p', amount=SNP_NOISE_AMOUNT))
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
                    im_float /= MAX_PIXEL_VALUE
                im_resized = resize(im_float, img_size, anti_aliasing=True)
                X.append(im_resized)
                y.append(label_idx)
                image_paths.append(path)
                if i_aug == 0:
                    original_indices.append(len(X)-1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, image_paths, original_indices

def save_display_image(img_array, label_text, save_path):
    """Save image with label and probability as high-resolution PNG."""
    img_array = np.clip(img_array, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(4,4), dpi=150)
    ax.imshow(img_array)
    ax.set_title(label_text, fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Build CNN
# ----------------------------
def build_cnn(input_shape=(128,128,3), num_classes=2):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(FILTERS_LAYER1, KERNEL_SIZE, activation='relu'),
        MaxPooling2D(),
        Conv2D(FILTERS_LAYER2, KERNEL_SIZE, activation='relu'),
        MaxPooling2D(),
        Conv2D(FILTERS_LAYER3, KERNEL_SIZE, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

    # Save model
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # ----------------------------
    # Results folder and file
    # ----------------------------
    RESULTS_DIR = "test_output"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_file_path = os.path.join(RESULTS_DIR, f"{script_name}_result.txt")
    f_out = open(output_file_path, "w")

    # Prediction on original images
    X_orig = X[original_indices]
    for idx, x_img in enumerate(X_orig):
        x_input = np.expand_dims(x_img, axis=0)
        prob = model.predict(x_input, verbose=0)[0]
        pred_idx = int(np.argmax(prob))
        label = CLASSES[pred_idx]
        label_text = f"{label} ({prob[pred_idx]:.2f})"

        # Save text
        line = f"{image_paths[original_indices[idx]]} → {label}, probability={prob[pred_idx]:.2f}\n"
        print(line, end="")
        f_out.write(line)

        # Save display image
        img_name = os.path.splitext(os.path.basename(image_paths[original_indices[idx]]))[0]
        save_path = os.path.join(RESULTS_DIR, f"{img_name}_{label}_tf.png")
        save_display_image(x_img, label_text, save_path)

        if not args.no_display:
            plt.imshow(x_img)
            plt.title(label_text)
            plt.axis('off')
            plt.show()

    # Optional external image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: {args.image} not found")
        else:
            img = imread(args.image)
            if img.ndim == 2:
                img = np.stack([img]*IMG_CHANNELS, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :IMG_CHANNELS]
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= MAX_PIXEL_VALUE
            img = resize(img, IMG_SIZE, anti_aliasing=True)
            x_input = np.expand_dims(img, axis=0)
            prob = model.predict(x_input, verbose=0)[0]
            pred_idx = int(np.argmax(prob))
            label = CLASSES[pred_idx]
            label_text = f"{label} ({prob[pred_idx]:.2f})"

            # Save text
            line = f"{args.image} → {label}, probability={prob[pred_idx]:.2f}\n"
            print(line, end="")
            f_out.write(line)

            # Save display image
            ext_img_name = os.path.splitext(os.path.basename(args.image))[0]
            save_path = os.path.join(RESULTS_DIR, f"{ext_img_name}_{label}_tf.png")
            save_display_image(img, label_text, save_path)

            if not args.no_display:
                plt.imshow(img)
                plt.title(label_text)
                plt.axis('off')
                plt.show()

    f_out.close()
    print(f"\nAll results saved to {output_file_path} and images saved in {RESULTS_DIR}/")
    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN crescent detection with data augmentation")
    parser.add_argument("--image", type=str, default=None, help="Optional external image to predict")
    parser.add_argument("--no-display", action="store_true", help="Disable display of images")
    args = parser.parse_args()
    main(args)

