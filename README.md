# Cat vs Dog Image Classifier

This project is a simple **Streamlit web app** that classifies uploaded images as either a **Cat** or a **Dog**, using a pre-trained Convolutional Neural Network (CNN) model built in TensorFlow/Keras.

##  Demo
Upload a `.jpg` or `.png` image and the app predicts whether it shows a cat or a dog, along with the model's confidence score.

## Folder Structure
```
cat-dog-streamlit/
├── model.py              # Streamlit app file
├── cat_dog_model.h5      # Trained CNN model (Keras)
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

## Features
- Upload image from your computer
- View the image in the browser
- Get predicted label and confidence score
- Powered by TensorFlow and Streamlit

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cat-dog-streamlit.git
cd cat-dog-streamlit
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the model
Place your trained `cat_dog_model.h5` file into the project folder.

### 4. Run the Streamlit app
```bash
streamlit run app.py

The app will open at `http://localhost:8501`
```
## Model Info
- CNN architecture trained on the Dogs vs Cats dataset (25,000 images)
- Input size: 150x150 pixels
- Output: Binary classification (0 = Cat, 1 = Dog)

##  Example Output
```
✅ Prediction: Dog 
🔍 Confidence: 0.93
```
## Author
- Soni Jain
- Model training done in Google Colab
- Streamlit frontend designed for interactive use
