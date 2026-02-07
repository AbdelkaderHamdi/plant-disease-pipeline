import streamlit as st
import cv2
import numpy as np
from .model_loader import list_models, load_model

def preprocess_image(uploaded_file, target_size=(64, 64)):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten().reshape(1, -1)

def predict(image, model, le):
    pred = model.predict(image)
    return le.inverse_transform(pred)[0]

st.set_page_config(page_title="Plant Disease Detection", layout="wide")
st.title("Plant Disease Detection Demo")

model_files = list_models(with_segmentation=False)
if not model_files:
    st.sidebar.error("No models found.")
    st.stop()

model_choice = st.sidebar.selectbox("Select Model", model_files)
model, le = load_model(model_choice, with_segmentation=False)
st.sidebar.success(f"Loaded model: {model_choice}")

uploaded_file = st.file_uploader(
    "Upload an image of a plant leaf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, use_column_width=True)
    image = preprocess_image(uploaded_file)
    prediction = predict(image, model, le)
    st.success(f"Predicted Disease: **{prediction}**")
