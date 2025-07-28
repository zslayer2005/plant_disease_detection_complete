# 🌿 Plant Disease Detection from Leaf Images

This project classifies tomato plant leaf diseases using a Convolutional Neural Network (CNN).

## 🔧 Tools Used
- Python
- TensorFlow + Keras
- OpenCV
- Streamlit
- PlantVillage Dataset (5 Tomato classes)

## 🚀 How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Launch web app:
```bash
streamlit run streamlit_app.py
```

## 📁 Dataset
Place 5 tomato disease folders under `/dataset/`:
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot
- Tomato_healthy

## model.h5
## This project includes a trained Convolutional Neural Network (CNN) model stored in the file:

Copy
Edit
model.h5
Purpose: This file contains the trained weights and architecture used for plant disease detection from leaf images.

-Framework: Built using Keras with TensorFlow backend.

-Training Data: Trained on the PlantVillage Dataset.

-Usage:

-Load this model in your Python/Streamlit app using:
-python
-Copy
-Edit
-from tensorflow.keras.models import load_model
-model = load_model('model.h5')
-Used to predict diseases from uploaded leaf images.

## 📦 Deliverables
- Trained CNN model
- Streamlit Web App
- Sample Dataset
