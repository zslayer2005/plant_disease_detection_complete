import numpy as np
import cv2
from tensorflow.keras.models import load_model

CLASS_NAMES = [
    "Tomato_Early_blight", 
    "Tomato_Late_blight", 
    "Tomato_Leaf_Mold", 
    "Tomato_Septoria_leaf_spot", 
    "Tomato_healthy"
]

MODEL_PATH = "model/plant_disease_model.h5"

def load_trained_model():
    return load_model(MODEL_PATH)

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image_class(model, processed_img):
    prediction = model.predict(processed_img)
    class_idx = np.argmax(prediction)
    return CLASS_NAMES[class_idx], float(np.max(prediction))
