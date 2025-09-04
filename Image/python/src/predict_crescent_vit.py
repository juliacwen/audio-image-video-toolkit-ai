#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Transformer Demo for Crescent Moon Detection
Author: Julia Wen
Date: 2025-09-03
Description:
    Fine-tunes a pre-trained Vision Transformer (ViT) on the crescent/no-crescent dataset.
    Supports prediction on original dataset images and optional external image.
    Displays color images with prediction probabilities.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
import argparse

# Dataset configuration
DATASET_DIR = "dataset"
CLASSES = ["crescent", "no_crescent"]
IMG_SIZE = 224  # ViT typically uses 224x224 input

# Argument parser
parser = argparse.ArgumentParser(description="Crescent Moon Prediction using ViT")
parser.add_argument("--image", type=str, help="Path to external image for prediction")
parser.add_argument("--no-display", action='store_true', help="Disable plotting images")
args = parser.parse_args()

# Custom dataset
class CrescentDataset(Dataset):
    def __init__(self, dataset_dir, classes, transform=None):
        self.image_paths, self.labels = [], []
        self.transform = transform
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(dataset_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, self.image_paths[idx]

# Data transforms (including augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# Load dataset
dataset = CrescentDataset(DATASET_DIR, CLASSES, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(CLASSES))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# Training loop
print(f"Training ViT on {len(dataset)} images...")
model.train()
for epoch in range(5):  # small number of epochs for demo
    running_loss = 0.0
    for imgs, labels, _ in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    print(f"Epoch {epoch+1}/5, Loss={running_loss/len(dataset):.4f}")

# Save model
torch.save(model.state_dict(), 'crescent_vit.pth')
print("ViT model saved to crescent_vit.pth")

# Evaluation on original images
model.eval()
softmax = nn.Softmax(dim=1)
for img, label, path in dataset:
    img_input = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_input)
        probs = softmax(outputs)
        pred_idx = probs.argmax(dim=1).item()
        prob = probs[0, pred_idx].item()
    print(f"{path} → {CLASSES[pred_idx]}, probability={prob:.2f}")
    if not args.no_display:
        plt.imshow(Image.open(path))
        plt.title(f"{os.path.basename(path)}: {CLASSES[pred_idx]} ({prob:.2f})")
        plt.axis('off')
        plt.show()

# Predict external image
if args.image:
    if os.path.exists(args.image):
        img = Image.open(args.image).convert('RGB')
        img_input = train_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_input)
            probs = softmax(outputs)
            pred_idx = probs.argmax(dim=1).item()
            prob = probs[0, pred_idx].item()
        print(f"\n{args.image} → {CLASSES[pred_idx]}, probability={prob:.2f}")
        if not args.no_display:
            plt.imshow(img)
            plt.title(f"{os.path.basename(args.image)}: {CLASSES[pred_idx]} ({prob:.2f})")
            plt.axis('off')
            plt.show()
    else:
        print(f"Error: {args.image} does not exist")

