"""
Pneumonia Detection from Chest X-rays with Explainable AI (Grad-CAM)

This Streamlit application provides:
1. Fast inference using TFLite model
2. Explainable predictions using Grad-CAM visualization
"""

import streamlit as st
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection with XAI",
    page_icon="🫁",
    layout="wide"
)

# Check if Keras model exists for Grad-CAM
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "resume_model.keras")
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.tflite")
GRADCAM_AVAILABLE = os.path.exists(KERAS_MODEL_PATH)


@st.cache_resource
def load_tflite_model():
    """Load TFLite model for fast inference."""
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_keras_model():
    """Load full Keras model for Grad-CAM."""
    if GRADCAM_AVAILABLE:
        from gradcam import load_keras_model as load_model
        return load_model(KERAS_MODEL_PATH)
    return None


def predict_tflite(interpreter, image):
    """Run prediction using TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get expected input shape
    input_shape = input_details[0]['shape']
    target_size = (input_shape[1], input_shape[2])
    
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    return prediction


def predict_with_gradcam(model, image):
    """Run prediction with Grad-CAM visualization."""
    from gradcam import get_gradcam_visualization
    return get_gradcam_visualization(model, image, layer_name="Conv4_3")


# App Header
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown("""
Upload a chest X-ray image to detect pneumonia with **AI-powered analysis** 
and **visual explanations** showing which regions influenced the decision.
""")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Mode selection
if GRADCAM_AVAILABLE:
    mode = st.sidebar.radio(
        "Analysis Mode",
        ["Fast (TFLite)", "Explainable (Grad-CAM)"],
        help="Grad-CAM provides visual explanations but is slower"
    )
else:
    mode = "Fast (TFLite)"
    st.sidebar.warning("⚠️ Keras model not found. Grad-CAM unavailable.")
    st.sidebar.info(f"Expected path: `{KERAS_MODEL_PATH}`")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This tool uses a CNN trained on chest X-ray images to detect pneumonia.

**Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights 
the regions that most influenced the AI's decision.

⚠️ **Disclaimer**: This is for educational purposes only. 
Always consult a medical professional.
""")

# Main content
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    if mode == "Fast (TFLite)":
        # Fast mode with TFLite
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Result")
            
            with st.spinner("Analyzing..."):
                interpreter = load_tflite_model()
                prediction = predict_tflite(interpreter, image)
            
            label = "Pneumonia" if prediction > 0.5 else "Normal"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Display result with color coding
            if label == "Pneumonia":
                st.error(f"### ⚠️ {label} Detected")
            else:
                st.success(f"### ✅ {label}")
            
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Confidence bar
            st.progress(float(confidence))
            
            st.info("💡 Switch to **Explainable (Grad-CAM)** mode for visual explanation")
    
    else:
        # Explainable mode with Grad-CAM
        st.subheader("🔬 Explainable AI Analysis")
        
        with st.spinner("Generating explanation... This may take a moment."):
            model = load_keras_model()
            label, confidence, heatmap_overlay, heatmap = predict_with_gradcam(model, image)
        
        # Three column layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original X-ray**")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("**Grad-CAM Heatmap**")
            st.image(heatmap_overlay, use_column_width=True)
        
        with col3:
            st.markdown("**Attention Map**")
            # Display raw heatmap
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(heatmap, cmap='jet')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Results
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            if label == "Pneumonia":
                st.error(f"### ⚠️ {label} Detected")
            else:
                st.success(f"### ✅ {label}")
            
            st.metric("Confidence", f"{confidence:.1%}")
        
        with result_col2:
            st.markdown("""
            ### 📊 Interpretation
            
            The **heatmap overlay** shows which regions of the X-ray the AI 
            focused on when making its prediction:
            
            - 🔴 **Red/Yellow areas**: High attention (most influential)
            - 🔵 **Blue areas**: Low attention (less influential)
            
            For pneumonia cases, the model typically focuses on:
            - Areas of lung opacity
            - Consolidation patterns
            - Abnormal lung markings
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    Pneumonia Detection with Explainable AI | 
    Built with TensorFlow & Streamlit |
    Model: Custom CNN with VGG16 Transfer Learning
    </small>
</div>
""", unsafe_allow_html=True)
