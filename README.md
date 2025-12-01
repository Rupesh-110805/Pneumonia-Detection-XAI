# 🫁 Pneumonia Detection with Explainable AI (XAI)

A deep learning project for detecting pneumonia from chest X-ray images using a custom CNN architecture with VGG16 transfer learning. Features **Grad-CAM** visualization for explainable AI predictions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

| Feature | Description |
|---------|-------------|
| **Task** | Binary Classification (Normal vs Pneumonia) |
| **Model** | Custom CNN with SeparableConv2D + BatchNormalization |
| **Transfer Learning** | VGG16 pretrained weights for early layers |
| **Input** | 224×224×3 RGB chest X-ray images |
| **Explainability** | Grad-CAM heatmap visualization |

## 📁 Project Structure

```
Pneumonia_detector/
├── app/                    # Streamlit application
│   ├── app.py              # Main Streamlit app
│   ├── gradcam.py          # Grad-CAM implementation
│   └── __init__.py
├── data/                   # Dataset (gitignored)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                 # Trained models (gitignored)
│   ├── resume_model.keras  # Full Keras model (for Grad-CAM)
│   └── model.tflite        # TFLite model (for fast inference)
├── notebooks/              # Jupyter notebooks
│   └── training.ipynb      # Model training notebook
├── docs/                   # Documentation
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Rupesh-110805/Pneumonia-Detection-XAI.git
cd Pneumonia-Detection-XAI
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Files
Download the trained models and place them in the `models/` folder:
- `resume_model.keras` - Full Keras model (required for Grad-CAM)
- `model.tflite` - TFLite model (for fast inference)

### 5. Run the Application
```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

## 🎯 Features

### Two Inference Modes

| Mode | Speed | Features |
|------|-------|----------|
| **⚡ Fast Mode** | ~50ms | TFLite inference, confidence score |
| **🔍 Explainable Mode** | ~500ms | Full prediction + Grad-CAM heatmap |

### Grad-CAM Visualization

The Explainable Mode generates a Grad-CAM heatmap showing which regions of the X-ray the model focused on:

- **Red/Yellow regions**: High importance for prediction
- **Blue/Green regions**: Low importance

## 🏗️ Model Architecture

```
Input (224×224×3)
    │
    ├── Conv2D Block 1 (64 filters) ← VGG16 weights
    ├── Conv2D Block 2 (64 filters) ← VGG16 weights
    │
    ├── SeparableConv2D Block 3 (128 filters)
    ├── SeparableConv2D Block 4 (256 filters) + BatchNorm
    ├── SeparableConv2D Block 5 (512 filters) + BatchNorm
    │
    ├── Flatten
    ├── Dense (1024) + Dropout (0.7)
    ├── Dense (512) + Dropout (0.5)
    └── Dense (1, sigmoid) → Output
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~92% |
| **Precision** | ~90% |
| **Recall** | ~95% |

## 📦 Dependencies

- TensorFlow 2.20+
- Streamlit 1.51+
- OpenCV (cv2)
- Pillow
- NumPy
- Matplotlib

## 🔬 Training

The model was trained on the Chest X-Ray Images (Pneumonia) dataset:
- **Training**: ~5,200 images
- **Validation**: ~16 images
- **Test**: ~624 images

See `notebooks/training.ipynb` for the complete training pipeline.

## 📄 License

MIT License - feel free to use this project for educational and research purposes.

## 👤 Author

**Rupesh** - [GitHub](https://github.com/Rupesh-110805)

## 🙏 Acknowledgments

- [Chest X-Ray Images Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- VGG16 pretrained weights from ImageNet
- Grad-CAM paper: [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1610.02391)
