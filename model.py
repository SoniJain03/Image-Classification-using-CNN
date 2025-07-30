import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
# type: ignore
import gdown

# Download the model if not present
model_path = "cat_dog_model.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1gszaN7ZOjE1Ljc1DRpo2TyGJ2KB-nOLW"
    gdown.download(url, model_path, quiet=False)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ["Cat", "Dog"]

# Title
st.title("🐱🐶 Cat vs Dog Image Classifier")
st.write("Upload an image, and the model will predict whether it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((180, 180))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 180, 180, 3)
    
    prediction = model.predict(image_array)
    predicted_class = class_names[int(prediction[0][0] > 0.5)]
    confidence = prediction[0][0] if predicted_class == "Dog" else 1 - prediction[0][0]

    st.success(f"Prediction: **{predicted_class}** with confidence {confidence * 100:.2f}%")
