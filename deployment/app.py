# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import cv2
import numpy as np
from .model_loader import list_models, latest_model

app = FastAPI(title="Plant Disease Detection API")

# Config
TARGET_SIZE = (64, 64)
WITH_SEGMENTATION = False

# Load latest model on startup
model, le = latest_model(with_segmentation=WITH_SEGMENTATION)

@app.get("/")
def home():
    return {"status": "Plant Disease Detection API is running"}

def preprocess_image(file_bytes, target_size=TARGET_SIZE):
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten().reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, le

    if model is None or le is None:
        raise HTTPException(status_code=500, detail="No model loaded")

    try:
        file_bytes = await file.read()
        image = preprocess_image(file_bytes)
        pred = model.predict(image)
        label = le.inverse_transform(pred)[0]
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def get_models():
    models = list_models(with_segmentation=WITH_SEGMENTATION)
    return {"models": models}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
