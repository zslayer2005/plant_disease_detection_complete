import streamlit as st
from app.utils import load_trained_model, preprocess_image, predict_image_class

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection from Leaf Image")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
model = load_trained_model()

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    processed_img = preprocess_image(uploaded_file)
    label, confidence = predict_image_class(model, processed_img)
    st.success(f"Prediction: **{label}** with confidence **{confidence*100:.2f}%**")
