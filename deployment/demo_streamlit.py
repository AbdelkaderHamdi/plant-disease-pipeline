import streamlit as st
import os
import cv2
import numpy as np
import joblib

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models/without_segmentation")  # Change if using segmentation

# Helper functions
def load_model(model_path, le_path):
    """Load model and label encoder"""
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le

def preprocess_image(uploaded_file, target_size=(64, 64)):
    """Preprocess uploaded image for prediction"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten().reshape(1, -1)

def predict(image, model, le):
    """Predict class of the image"""
    pred = model.predict(image)
    label = le.inverse_transform(pred)[0]
    return label

# Streamlit UI
st.set_page_config(page_title="Plant Disease Detection", layout="wide")
st.title("Plant Disease Detection Demo")

# Sidebar: model selection
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib") and "_le_" not in f]
model_choice = st.sidebar.selectbox("Select Model", model_files)

# Automatically load the corresponding label encoder
le_file = model_choice.replace(".joblib", "_le.joblib")
model_path = os.path.join(MODELS_DIR, model_choice)
le_path = os.path.join(MODELS_DIR, le_file)

model, le = load_model(model_path, le_path)
st.sidebar.success(f"Loaded model: {model_choice}")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    st.write("Processing image...")
    image = preprocess_image(uploaded_file)
    prediction = predict(image, model, le)
    
    st.success(f"Predicted Disease: **{prediction}**")
