import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

@st.cache_resource
def load_model():
    model_path = "cat_dog_model.h5"
    
    # Download the model if not already downloaded
    if not os.path.exists(model_path):
        gdown.download(
            "https://drive.google.com/uc?id=1gszaN7ZOjE1Ljc1DRpo2TyGJ2KB-nOLW",
            model_path,
            quiet=False
        )

    # Load and return model
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ["Cat", "Dog"]

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload a cat or dog image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((180, 180))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 180, 180, 3)

    prediction = model.predict(image_array)
    predicted_class = class_names[int(prediction[0][0] > 0.5)]
    confidence = prediction[0][0] if predicted_class == "Dog" else 1 - prediction[0][0]

    st.success(f"Prediction: **{predicted_class}** ({confidence * 100:.2f}% confidence)")
