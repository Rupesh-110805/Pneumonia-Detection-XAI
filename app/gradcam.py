"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
For Explainable AI in Pneumonia Detection

This module generates visual explanations for CNN predictions by highlighting
the regions of the input image that are most important for the prediction.
"""

import numpy as np
import tensorflow as tf
import cv2
keras = tf.keras # type: ignore

class GradCAM:
    """
    Grad-CAM implementation for generating visual explanations.
    
    Attributes:
        model: Keras model for inference
        layer_name: Name of the convolutional layer to use for Grad-CAM
    """
    
    def __init__(self, model, layer_name="Conv4_3"):
        """
        Initialize GradCAM with a model and target layer.
        
        Args:
            model: Loaded Keras model
            layer_name: Name of the conv layer to extract gradients from
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        """Build a model that outputs both predictions and conv layer activations."""
        return keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
    
    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap for the given image.
        
        Args:
            img_array: Preprocessed image array (1, H, W, 3)
            pred_index: Index of the target class (None for predicted class)
            
        Returns:
            heatmap: Normalized heatmap array
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            
            # Convert to tensor if needed
            if isinstance(predictions, list):
                predictions = predictions[0]
            predictions = tf.convert_to_tensor(predictions)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])  # type: ignore
            
            # For binary classification with sigmoid
            if len(predictions.shape) == 1 or predictions.shape[-1] == 1:
                class_output = predictions[0, 0] if len(predictions.shape) > 1 else predictions[0] # type: ignore
            else:
                class_output = predictions[:, pred_index] # type: ignore
        
        # Compute gradients of the class output with respect to conv layer
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay the heatmap on the original image.
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original input image (PIL Image or numpy array)
            alpha: Transparency for the overlay
            colormap: OpenCV colormap for visualization
            
        Returns:
            superimposed_img: Image with heatmap overlay
        """
        # Convert PIL Image to numpy if needed
        if hasattr(original_image, 'convert'):
            img = np.array(original_image.convert('RGB'))
        else:
            img = original_image
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), colormap # type: ignore
        ) # type: ignore
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose the heatmap on original image
        superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)
        
        return superimposed_img


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        target_size: Target size for resizing
        
    Returns:
        img_array: Preprocessed numpy array ready for model input
    """
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_keras_model(model_path):
    """
    Load the Keras model for Grad-CAM.
    
    Args:
        model_path: Path to the .keras or .h5 model file
        
    Returns:
        model: Loaded Keras model
    """
    return keras.models.load_model(model_path)


def get_gradcam_visualization(model, image, layer_name="Conv4_3"):
    """
    Complete pipeline to generate Grad-CAM visualization.
    
    Args:
        model: Keras model
        image: PIL Image
        layer_name: Target conv layer name
        
    Returns:
        tuple: (prediction, confidence, heatmap_overlay)
    """
    # Preprocess
    img_array = preprocess_image(image)
    
    # Get prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, layer_name)
    heatmap = gradcam.generate_heatmap(img_array)
    heatmap_overlay = gradcam.overlay_heatmap(heatmap, image)
    
    return label, confidence, heatmap_overlay, heatmap
