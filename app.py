import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="CNN Image Classifier", layout="centered")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cat_dog_model.h5")  # Make sure model.h5 is in same folder
    return model

model = load_model()

# ---------------------- CLASS NAMES ----------------------
class_names = ['Cat', 'Dog']  # Update this list with your actual class labels

# ---------------------- PREPROCESS FUNCTION ----------------------
def preprocess_image(image):
    image = image.resize((150, 150))  # Match model input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ---------------------- SIDEBAR (ABOUT SECTION) ----------------------
st.sidebar.markdown("## 🧠 About This App")
st.sidebar.markdown(
    """
Welcome to the **CNN Image Classifier**!  
This app uses a trained **Convolutional Neural Network** to classify images of **Cats and Dogs**.

---

### 🔍 How It Works:
- Upload a JPG or PNG image.
- The model will analyze the image.
- You'll get the **predicted class** and a **confidence score**.

---

### 🛠️ Built With:
- Python 🐍  
- Streamlit 🎈  
- TensorFlow ⚙️  
- Pillow 🖼️ + NumPy 🔢  

---

👨‍💻 *Developed by Soni Jain*  
📁 Model trained locally and loaded from `.h5` file.
"""
)

# ---------------------- MAIN PAGE ----------------------
st.title("🐾 CNN Image Classifier")
st.write("Upload an image and the model will predict whether it's a **Cat** or a **Dog**.")

# ---------------------- IMAGE UPLOADER ----------------------
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("🔎 Classifying..."):
        preprocessed_img = preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_class = class_names[predicted_index]

    # Output
    st.success(f"### 🧠 Prediction: `{predicted_class}`")
    st.info(f"🔢 Confidence Score: **{confidence:.2f}**")

    # Detailed prediction scores
    st.subheader("📊 Prediction Scores:")
    for idx, prob in enumerate(predictions[0]):
        st.write(f"- **{class_names[idx]}**: {prob:.4f}")
else:
    st.warning("👈 Please upload an image to get started.")
