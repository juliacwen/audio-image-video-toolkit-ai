# File: Image/python/src/server.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — FastAPI wrapper for crescent moon detection
Author: Julia Wen (wendigilane@gmail.com)
Date: 2025-10-15

Purpose:
--------
Expose the HOG + SVM crescent detector (from predict_crescent_pytorch.py)
as a REST API so that a TypeScript/React frontend can send images for
prediction and receive real results.

File placement:
---------------
This file should be located under:
    Image/python/src/server.py

And the model script should be:
    Image/python/src/predict_crescent_pytorch.py

Usage:
------
1. Make sure the following files are trained and present:
   - crescent_moon_model.pkl
   - crescent_moon_scaler.pkl
2. Install dependencies:
   pip install fastapi uvicorn python-multipart joblib scikit-image scikit-learn numpy
3. Run the API server (from the src directory):
   cd Image/python/src
   uvicorn server:app --reload
4. The endpoint:
   POST /detect
   Input: multipart/form-data with one uploaded image file
   Output JSON: { "detected": bool, "confidence": float, "filename": str }
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
import joblib
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
from predict_crescent_pytorch import HOG_PARAMS, IMG_SIZE, CLASSES

SCALER_PATH = "crescent_moon_scaler.pkl"
MODEL_PATH = "crescent_moon_model.pkl"

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

