import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Load and cache model
@st.cache_resource
def load_model():
    model_path = "cat_dog_model.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1gszaN7ZOjE1Ljc1DRpo2TyGJ2KB-nOLW"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ["Cat", "Dog"]

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((180, 180))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 180, 180, 3)

    prediction = model.predict(img_array)[0][0]
    predicted_class = class_names[int(prediction > 0.5)]
    confidence = prediction if predicted_class == "Dog" else 1 - prediction

    st.success(f"Prediction: **{predicted_class}** ({confidence * 100:.2f}% confidence)")
