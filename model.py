import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ This MUST be the first Streamlit command
st.set_page_config(page_title="Cat vs Dog Classifier ")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_model.h5")

model = load_model()

st.title(" Image Classifaction: Cats and Dogs using CNN ")
st.markdown("Upload an image to check whether it's a **Cat** or a **Dog**.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Dog " if prediction > 0.5 else "Cat "
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}`")

# 📌 About section in sidebar
with st.sidebar:
    st.header("📖 About")
    st.markdown("""
This app uses a **Convolutional Neural Network (CNN)** to classify images of cats 🐱 and dogs 🐶.

- Built using **TensorFlow** and **Streamlit**
- Trained on the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats)
- Developed by **Soni Jain** ✨

Upload a `.jpg` or `.png` image and get an instant prediction!
    """)
