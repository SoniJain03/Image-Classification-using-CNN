import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Download & cache the model from Google Drive
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

# Load the model
model = load_model()
class_names = ["Cat", "Dog"]

# Title and description
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog and the model will predict which it is.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((180, 180))  # Ensure 3 channels
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 180, 180, 3)

    # Predict
    prediction = model.predict(img_array)[0][0]
    predicted_class = class_names[int(prediction > 0.5)]
    confidence = prediction if predicted_class == "Dog" else 1 - prediction

    # Display result
    st.success(f"Prediction: **{predicted_class}** ({confidence * 100:.2f}% confidence)")
