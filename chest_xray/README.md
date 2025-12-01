# 🫁 Pneumonia Detection from Chest X-ray using Deep Learning

A web-based application that uses a **Convolutional Neural Network (CNN)** model to detect **Pneumonia** from chest X-ray images. Built with TensorFlow, converted to TFLite for lightweight deployment, and served using **Streamlit**.

## 🚀 Live Demo

👉 [Click here to try the app](https://chestxray-<your-app-id>.streamlit.app/)  
*(Replace with your actual deployed Streamlit Cloud link)*

---

## 📌 Features

- Upload chest X-ray images directly through the browser
- Real-time prediction: "Pneumonia" or "Normal"
- Model optimized with TensorFlow Lite for efficient inference
- Streamlit-powered user interface
- Fully deployed and accessible online

---

## 🧠 Model Architecture

- **Input Shape**: `(224, 224, 3)`
- **CNN Layers**: Depthwise Separable Convolutions + BatchNorm
- **Pooling**: MaxPooling
- **Fully Connected Layers**
- **Output**: Binary classification (Sigmoid activation)

---

## 📂 Project Structure

chest_xray/
│
├── app.py # Streamlit application
├── requirements.txt # Dependencies for deployment
├── .gitignore
├── model/
│ └── model.tflite # Trained & quantized TFLite model


---

## 🖼 Sample Predictions

| Image | Prediction |
|-------|------------|
| ![](sample_normal.png) | ✅ Normal |
| ![](sample_pneumonia.png) | ❗ Pneumonia |

---

## 🧪 How to Run Locally

1. Clone the repo:
   ```bash
   cd chest_xray

⚙️ Requirements
Python 3.9+

TensorFlow Lite Runtime

Streamlit

NumPy, Pillow, etc.

(All dependencies are listed in requirements.txt)

🏥 Dataset
The model was trained on a subset of the Chest X-ray dataset from Kaggle, containing:

Normal X-rays

Pneumonia-infected X-rays

📊 Model Performance
Metric	Score
Accuracy	~95%
Precision	High
Recall	High
Loss	Very Low

Model evaluation was done on a separate validation set.

📦 Deployment
This app is deployed using Streamlit Cloud, and the TFLite model allows for faster, lightweight inference.



🛡 Disclaimer
This tool is intended for educational purposes only and should not be used for medical diagnosis.

🌟 Show Your Support
If you like this project, leave a ⭐ on the repo or connect with me on LinkedIn.


---

