#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_crescent_pytorch_cnn.py
Author: Julia Wen 
Date: 2025-09-08 Adapted to CNN
Date: 2025-09-18 Updated to outpout to json
Description:
End-to-end CNN for crescent classification:
 - Lightweight CNN (2 conv layers + pooling + dropout + FC head)
 - Train/validation split with metrics tracking
 - Augmentation for training
 - Plots training vs validation curves and saves as PNG
 - Predict on original dataset and optional new images
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
from skimage.util import random_noise
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# -------------------
# Config
# -------------------
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
MODEL_PATH = "test_output/crescent_cnn_best.pth"

# -------------------
# Dataset
# -------------------
def augment_image_list(img_color):
    imgs = [img_color]
    imgs.append(np.fliplr(img_color))
    for angle in (-15, 15):
        imgs.append(rotate(img_color, angle, resize=False, mode='edge'))
    imgs.append(random_noise(img_color, mode='s&p', amount=0.01))
    return imgs

class CrescentDataset(Dataset):
    def __init__(self, dataset_dir, classes, img_size, augment=True):
        self.samples = []
        for label_idx, cls in enumerate(classes):
            class_dir = os.path.join(dataset_dir, cls)
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
                for im_color in imgs_to_use:
                    imf = im_color.astype(np.float32)
                    if imf.max() > 1.0:
                        imf /= 255.0
                    im_resized = resize(imf, img_size, anti_aliasing=True)
                    # convert to CHW tensor
                    im_chw = np.transpose(im_resized, (2, 0, 1))
                    self.samples.append((im_chw, label_idx, path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        im_chw, label, path = self.samples[idx]
        return torch.tensor(im_chw, dtype=torch.float32), label, path

# -------------------
# Model
# -------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * (IMG_SIZE[0]//4) * (IMG_SIZE[1]//4), 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# -------------------
# Training + Validation
# -------------------
def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).long()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        # Compute averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Track best val accuracy and save model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f} "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}"
        )

    # --- Plot curves ---
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(10,4))

    # Loss plot
    plt.subplot(1,2,1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1,2,2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    print(f"\n✅ Best Validation Accuracy: {best_val_acc:.2f} at epoch {best_epoch}")
    print(f"✅ Best model saved to {MODEL_PATH}")

    return history, best_val_acc, best_epoch

# -------------------
# Prediction
# -------------------

def predict(model, dataset, device, no_display=False):
    model.eval()
    crescent_probs = {}  # store probability of "crescent" class

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label, path = dataset[idx]
            img = img.unsqueeze(0).to(device)
            logits = model(img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            pred_name = CLASSES[pred_idx]
            crescent_probs[path] = float(probs[CLASSES.index("crescent")])  # store crescent probability

            print(f"{path} → {pred_name}, probability={probs[pred_idx]:.2f}")
            if not no_display:
                plt.imshow(imread(path))
                plt.title(f"{os.path.basename(path)}: {pred_name} ({probs[pred_idx]:.2f})")
                plt.axis("off")
                plt.show()

    # Save all crescent probabilities to JSON
    json_path = "test_output/crescent_probs.json"
    with open(json_path, "w") as f:
        json.dump(crescent_probs, f, indent=4)
    print(f"\n✅ Crescent probabilities saved to {json_path}")

    return crescent_probs


# -------------------
# Main
# -------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset (with augmentation)
    full_dataset = CrescentDataset(DATASET_DIR, CLASSES, IMG_SIZE, augment=True)

    # Split into train/val
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Train CNN with validation
    model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    train_model(model, train_loader, val_loader, EPOCHS, LR, device)

    # Predict on original images (no augmentation)
    test_dataset = CrescentDataset(DATASET_DIR, CLASSES, IMG_SIZE, augment=False)
    print("\nPrediction summary (original images):")
    predict(model, test_dataset, device, args.no_display)

    # -------------------
    # Optional: predict new image with temperature scaling
    # -------------------
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: {args.image} not found")
            return
        img = imread(args.image)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        imf = img.astype(np.float32)
        if imf.max() > 1.0:
            imf /= 255.0
        im_resized = resize(imf, IMG_SIZE, anti_aliasing=True)
        im_chw = np.transpose(im_resized, (2, 0, 1))
        tensor = torch.tensor(im_chw, dtype=torch.float32).unsqueeze(0).to(device)

        TEMPERATURE = 1.5  # <--- only scales new/unseen image

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits / TEMPERATURE, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        pred_name = CLASSES[pred_idx]
        print(f"\n{args.image} → {pred_name}, probability={probs[pred_idx]:.2f}")

        if not args.no_display:
            plt.imshow(img)
            plt.title(f"{os.path.basename(args.image)}: {pred_name} ({probs[pred_idx]:.2f})")
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN for crescent classification")
    parser.add_argument("--image", type=str, help="Optional new image")
    parser.add_argument("--no-display", action="store_true", help="Disable image display")
    args = parser.parse_args()
    main(args)

