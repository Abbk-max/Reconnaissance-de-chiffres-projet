import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from rembg import remove, new_session
import streamlit as st

# ==============================================================================
# 1. MODEL & RESOURCE MANAGEMENT
# ==============================================================================

@st.cache_resource
def load_trained_model(model_path="mnist_model.keras"):
    """Loads the compiled Keras model from disk."""
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def get_rembg_session():
    """Initializes and caches the U2NETP session for background removal."""
    return new_session("u2netp")

@st.cache_data
def load_mnist_test_data():
    """Loads a subset of MNIST data for testing purposes."""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test

# ==============================================================================
# 2. IMAGE PREPROCESSING PIPELINE
# ==============================================================================

def process_image_pipeline(image_pil):
    """
    Main pipeline to transform a raw user image into a model-ready tensor.
    
    Args:
        image_pil (PIL.Image): The input image.
        
    Returns:
        tuple: (processed_tensor, quality_score)
               - processed_tensor: (1, 28, 28, 1) float32 array
               - quality_score: float (0.0 to 1.0) indicating detection confidence.
    """
    # 1. Standardization: Ensure RGB
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
        
    try:
        # 2. Background Removal & Grayscale
        session = get_rembg_session()
        img_no_bg = remove(image_pil, session=session)
        alpha = np.array(img_no_bg)[:, :, 3] # Extract Alpha channel
        
        img_gray = np.array(ImageOps.grayscale(image_pil))
        
        # 3. Denoising
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # 4. Smart Inversion (Ensure white text on black background)
        # We check the brightness of the object defined by the alpha mask
        mask_obj = alpha > 10
        if np.sum(mask_obj) > 0:
            mean_val = np.mean(img_blurred[mask_obj])
            if mean_val < 127: # Dark digit on light paper
                img_blurred = 255 - img_blurred
        
        # 5. Apply Mask (Isolate digit)
        img_masked = (img_blurred.astype(float) * (alpha.astype(float) / 255.0)).astype(np.uint8)
        
    except Exception as e:
        print(f"Preprocessing Fallback: {e}")
        # Fallback: Simple inversion if advanced background removal fails
        img_masked = np.array(ImageOps.invert(image_pil.convert('L')))

    # 6. Localization (Finding the digit)
    _, binary = cv2.threshold(img_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 2
        # Safe slicing with padding
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(img_masked.shape[1] - x, w + 2*pad)
        h = min(img_masked.shape[0] - y, h + 2*pad)
        roi = img_masked[y:y+h, x:x+w]
        quality = 1.0
    else:
        roi = img_masked
        quality = 0.0

    # 7. Resize to 20x20 (preserving aspect ratio)
    roi_h, roi_w = roi.shape
    if roi_h > 0 and roi_w > 0:
        scale = 20.0 / max(roi_h, roi_w)
        new_h, new_w = int(roi_h * scale), int(roi_w * scale)
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        roi_resized = np.zeros((20, 20), dtype=np.uint8)

    # 8. Center in 28x28 Canvas (Center of Mass)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    final_img[start_y:start_y+new_h, start_x:start_x+new_w] = roi_resized
    
    moments = cv2.moments(final_img)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        shift_x, shift_y = 14 - cx, 14 - cy
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        final_img = cv2.warpAffine(final_img, M, (28, 28))
        
    # 9. Normalize & Reshape
    final_img = cv2.normalize(final_img, None, 0, 255, cv2.NORM_MINMAX)
    img_tensor = final_img.astype('float32') / 255.0
    img_tensor = img_tensor.reshape(1, 28, 28, 1)
    
    return img_tensor, quality

# ==============================================================================
# 3. PREDICTION LOGIC
# ==============================================================================

def predict_digit(model, img_tensor, tta_steps=5):
    """
    Performs prediction using Test Time Augmentation (TTA) for robustness.
    """
    preds = []
    # Original prediction
    preds.append(model.predict(img_tensor, verbose=0))
    
    # Augmented predictions (slight shifts)
    for _ in range(tta_steps):
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        shifted = np.roll(img_tensor, shift_x, axis=1)
        shifted = np.roll(shifted, shift_y, axis=2)
        preds.append(model.predict(shifted, verbose=0))

    # Average predictions
    avg_pred = np.mean(preds, axis=0)
    return avg_pred
