import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import gdown

# ---- Load Model ----
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

model = load_model()
class_names = ["Cat", "Dog"]

# ---- Sidebar Info ----
st.sidebar.title("📘 About")
st.sidebar.markdown("""
This is a **deep learning** model built using **CNN** to classify images of cats and dogs.

- 🧠 Framework: TensorFlow / Keras  
- 📊 Trained on: 25,000+ labeled images  
- 🐾 Input: JPG or PNG image  
- ☁️ Model hosted via Google Drive  
""")

# ---- Main App UI ----
st.title("🐱🐶 Cat vs Dog Classifier")
st.markdown("Upload an image of a **cat** or **dog**, and the model will predict which it is!")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---- Preprocessing ----
    image = ImageOps.fit(image, (180, 180), Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0

    if img_array.shape != (180, 180, 3):
        st.error("Image shape is incompatible. Please upload a valid RGB image.")
    else:
        img_array = img_array.reshape(1, 180, 180, 3)

        # ---- Prediction ----
        prediction = model.predict(img_array)[0][0]
        predicted_class = class_names[int(prediction > 0.5)]
        confidence = prediction if predicted_class == "Dog" else 1 - prediction

        st.success(f"**Prediction:** {predicted_class} ({confidence * 100:.2f}% confidence)")
