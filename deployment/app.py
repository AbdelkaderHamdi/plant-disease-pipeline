# app.py
from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from model_loader import load_model, list_models

app = Flask(__name__)

# Configuration
TARGET_SIZE = (64, 64)  # Match preprocessing in training
WITH_SEGMENTATION = False  # Change if using segmentation models

# Load latest model on startup
model, le = load_model(*list(load_model(latest_model_name := None if WITH_SEGMENTATION==False else None))) if False else (None, None)

@app.route('/')
def home():
    return "🌱 Plant Disease Detection API is running!"

def preprocess_image(file, target_size=TARGET_SIZE):
    """Convert uploaded image to model input"""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten().reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    global model, le
    if model is None or le is None:
        return jsonify({"error": "No model loaded. Please load a model first."}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        image = preprocess_image(file)
        pred = model.predict(image)
        label = le.inverse_transform(pred)[0]
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Return list of available models"""
    models = list_models(with_segmentation=WITH_SEGMENTATION)
    return jsonify({"models": models})

if __name__ == '__main__':
    # Load the latest model dynamically
    from model_loader import latest_model
    model, le = latest_model(with_segmentation=WITH_SEGMENTATION)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
