#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_predict_crescent_tf.py
Author: Julia Wen
Date: 2025-09-20
Description:
Streamlit app for TensorFlow CNN crescent detection.
Supports live training from scratch or loading existing model.
Data augmentation can be toggled. Prediction with upload image.
Saves labeled `_tf.png` images and results text in `test_output/`.
Revision History:
2025-09-20 — Added live training, augmentation toggle, upload prediction, results saving.
2025-09-21 — Refactored to remove all magic numbers; constants centralized.
2025-09-21 — Fixed imsave TypeError and added explicit training workflow.
"""

import os
import numpy as np
import streamlit as st
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import resize, rotate
from skimage.util import random_noise
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

# ----------------------------
# Constants / Config
# ----------------------------
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = (128, 128)
MODEL_PATH = "crescent_moon_cnn.keras"
TEST_OUTPUT_DIR = "test_output"
RNG_SEED = 42
AUG_ANGLES = (-15, 15)
NOISE_AMOUNT = 0.01
CONV_FILTERS = [32, 64, 128]
KERNEL_SIZE = (3, 3)
DENSE_UNITS = 128
DROPOUT_RATE = 0.5
np.random.seed(RNG_SEED)

# Ensure output directory exists
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def augment_color_image_list(img_color):
    imgs = [img_color]
    try:
        imgs.append(np.fliplr(img_color))
    except Exception:
        imgs.append(img_color.copy())
    for angle in AUG_ANGLES:
        imgs.append(rotate(img_color, angle, resize=False, mode='edge'))
    imgs.append(random_noise(img_color, mode='s&p', amount=NOISE_AMOUNT))
    return imgs

def load_images(dataset_dir, classes, img_size, augment=True):
    X, y, image_paths, original_indices = [], [], [], []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
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

# ----------------------------
# CNN Model
# ----------------------------
def build_cnn(input_shape=(128,128,3), num_classes=2, lr=1e-3):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(CONV_FILTERS[0], KERNEL_SIZE, activation='relu'),
        MaxPooling2D(),
        Conv2D(CONV_FILTERS[1], KERNEL_SIZE, activation='relu'),
        MaxPooling2D(),
        Conv2D(CONV_FILTERS[2], KERNEL_SIZE, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# Streamlit App
# ----------------------------
st.title("Crescent Moon Detection (TensorFlow CNN)")

# Sidebar parameters
st.sidebar.header("Training Parameters")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.slider("Batch Size", min_value=2, max_value=32, value=8)
lr = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")
use_augmentation = st.sidebar.checkbox("Use Data Augmentation", value=True)

# Dataset input
dataset_dir = st.text_input("Dataset Directory", DATASET_DIR)
train_button = st.button("Start Training")

# Train model if user requests
model = None
if os.path.exists(MODEL_PATH):
    st.info(f"Loading existing model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
elif train_button:
    X, y, _, _ = load_images(dataset_dir, CLASSES, IMG_SIZE, augment=use_augmentation)
    if len(X) == 0:
        st.error("No images found in dataset directory.")
    else:
        model = build_cnn(input_shape=X.shape[1:], num_classes=len(CLASSES), lr=lr)
        st.info("Training started...")
        model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)
        model.save(MODEL_PATH)
        st.success(f"Training finished and model saved to {MODEL_PATH}")

# Image prediction
st.header("Predict Image")
uploaded_file = st.file_uploader("Upload an image", type=['png','jpg','jpeg','bmp','tif','tiff'])
if uploaded_file and model:
    img = imread(uploaded_file)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    im_float = img.astype(np.float32)
    if im_float.max() > 1.0:
        im_float /= 255.0
    im_resized = resize(im_float, IMG_SIZE, anti_aliasing=True)
    x_input = np.expand_dims(im_resized, axis=0)
    prob = model.predict(x_input, verbose=0)[0]
    pred_idx = int(np.argmax(prob))
    label = CLASSES[pred_idx]
    st.image(im_resized, caption=f"Prediction: {label} ({prob[pred_idx]:.2f})")

    # Save labeled image as uint8
    save_name = os.path.splitext(os.path.basename(uploaded_file.name))[0] + "_tf.png"
    save_path = os.path.join(TEST_OUTPUT_DIR, save_name)
    imsave(save_path, img_as_ubyte(np.clip(im_resized, 0.0, 1.0)))

    # Append result to results file
    results_file = os.path.join(TEST_OUTPUT_DIR, "predict_crescent_tf_result.txt")
    with open(results_file, "a") as f:
        f.write(f"{uploaded_file.name} → {label}, probability={prob[pred_idx]:.2f}\n")
    st.success(f"Results saved to {results_file} and labeled image saved to {save_path}")

