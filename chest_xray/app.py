import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model/model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize((150, 150))  
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction

st.title("Pneumonia Detection from Chest X-rays")
st.markdown("Upload a chest X-ray image and get a prediction: **Pneumonia** or **Normal**.")

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    interpreter = load_model()
    output = predict(interpreter, image)

    label = "Pneumonia" if output > 0.5 else "Normal"
    st.subheader(f"Prediction: **{label}**")
    st.text(f"Confidence Score: {output:.4f}")
