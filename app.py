import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --------- Page Config ---------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

# --------- Sidebar Info ---------
st.sidebar.title("📘 About the Project")
st.sidebar.markdown("""
Welcome to the **Cat vs Dog Classifier** 🐶🐱

This project uses a **CNN model** built with TensorFlow/Keras to predict whether an uploaded image is of a **cat** or **dog**.

### 💡 Features:
- Pre-trained on 25,000+ labeled images.
- Image input size: **180x180**
- Accurate prediction with confidence score.
- Deployed on **Streamlit Cloud**.

### 🔍 How to Use:
1. Upload an image of a cat or dog.
2. The model will process and classify the image.
3. Get prediction + confidence.

Made with ❤️ by Soni Jain
""")

# --------- Model Setup ---------
MODEL_PATH = "cat_dog_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1gszaN7ZOjE1Ljc1DRpo2TyGJ2KB-nOLW"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------- Prediction Function ---------
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((180, 180))  # ✅ Crucial step
    image = np.array(image) / 255.0
    return image.reshape(1, 180, 180, 3)

def predict(image: Image.Image) -> str:
    input_array = preprocess(image)
    prediction = model.predict(input_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if label == "Dog" else 1 - prediction
    return f"**Prediction:** {label} ({confidence * 100:.2f}% confidence)"

# --------- Main App ---------
st.title("🐾 Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Upload an image of a cat or dog:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Predicting..."):
        result = predict(image)
        st.success(result)
