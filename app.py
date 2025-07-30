import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import gdown

# Download and cache model
@st.cache_resource
def load_model():
    model_path = "cat_dog_model.h5"
    if not os.path.exists(model_path):
        gdown.download(
            "https://drive.google.com/uc?id=1gszaN7ZOjE1Ljc1DRpo2TyGJ2KB-nOLW",
            model_path,
            quiet=False
        )
    return tf.keras.models.load_model(model_path)

# Load model
model = load_model()

# Class labels
class_names = ["Cat", "Dog"]

# Sidebar "About" section
st.sidebar.title("📘 About")
st.sidebar.info(
    """
    **Cat vs Dog Classifier**  
    This project uses a Convolutional Neural Network (CNN) trained on a dataset of 25,000 images.  
    Upload an image of a **cat or dog**, and the model will predict the species with confidence.
    
    **Tech stack**: TensorFlow · Streamlit · Python  
    Model is hosted via Google Drive.
    """
)

# Main app
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and get an instant prediction!")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# When file is uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = ImageOps.fit(image, (180, 180), Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 180, 180, 3)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    predicted_class = class_names[int(prediction > 0.5)]
    confidence = prediction if predicted_class == "Dog" else 1 - prediction

    # Display result
    st.success(f"Prediction: **{predicted_class}** ({confidence * 100:.2f}% confidence)")
