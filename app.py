import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model_path = 'C:\Users\Boniface\Desktop\Malaria-Detection-App\Malaria_Model.h5'  
model = load_model(model_path)

# Define function for malaria detection
def detect_malaria(image_file):
    img = image.load_img(image_file, target_size=(68, 68))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    if result[0] == 0:
        return "Malaria Detected!"
    else:
        return "No Malaria Detected!"

# Streamlit app UI
st.title('Malaria Detection App')

uploaded_file = st.file_uploader("Upload an image of a cell", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    prediction = detect_malaria(uploaded_file)
    st.write(f"Prediction: {prediction}")
