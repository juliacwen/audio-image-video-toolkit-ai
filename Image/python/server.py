# File: Image/python/server.py
"""
File: Image/python/server.py
Description: FastAPI server for crescent moon detection in images.
Author: Julia Wen (wendigilane@gmail.com)
Date: 2025-11-18
Notes:
    - Receives image uploads via POST /detect.
    - Processes images using HOG features and a pre-trained SVM model.
    - Returns JSON with detection result, confidence, and filename.
    - Requires trained model and scaler files:
        - src/crescent_moon_model.pkl
        - src/crescent_moon_scaler.pkl
    - Uploads are stored temporarily in the 'uploads' folder.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALER_PATH = os.path.join(BASE_DIR, "src", "crescent_moon_scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "src", "crescent_moon_model.pkl")

from src.predict_crescent_pytorch import HOG_PARAMS, IMG_SIZE, CLASSES

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
import joblib
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import numpy as np


scaler = joblib.load(SCALER_PATH)
svm = joblib.load(MODEL_PATH)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load image
    img = imread(file_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    # HOG features
    im_gray = rgb2gray(resize(img, IMG_SIZE, anti_aliasing=True))
    feats = hog(im_gray, **HOG_PARAMS).astype(np.float32)
    feats_scaled = scaler.transform([feats])

    # Predict
    prob = svm.predict_proba(feats_scaled)[0]
    pred_idx = np.argmax(prob)
    label_name = CLASSES[pred_idx]

    return {"detected": label_name == "crescent", "confidence": float(prob[pred_idx]), "filename": file.filename}

