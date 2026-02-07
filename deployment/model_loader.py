# model_loader.py
import os
import joblib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")

def list_models(with_segmentation=False):
    """List all saved models"""
    folder = "with_segmentation" if with_segmentation else "without_segmentation"
    model_path = os.path.join(MODELS_DIR, folder)
    
    if not os.path.exists(model_path):
        return []
    
    # Only return model files (exclude label encoders and metrics)
    models = [f for f in os.listdir(model_path) if f.endswith(".joblib") and "_le_" not in f]
    return models

def load_model(model_name, with_segmentation=False):
    """Load a trained model and its label encoder"""
    folder = "with_segmentation" if with_segmentation else "without_segmentation"
    model_path = os.path.join(MODELS_DIR, folder, model_name)
    
    # Corresponding label encoder
    le_name = model_name.replace(".joblib", "_le.joblib")
    le_path = os.path.join(MODELS_DIR, folder, le_name)
    
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        raise FileNotFoundError(f"Model or label encoder not found: {model_name}")
    
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    return model, le

def latest_model(with_segmentation=False):
    """Return the latest trained model based on timestamp"""
    models = list_models(with_segmentation)
    if not models:
        return None, None
    # Sort by timestamp in filename
    models_sorted = sorted(models, reverse=True)
    latest_model_name = models_sorted[0]
    return load_model(latest_model_name, with_segmentation)
