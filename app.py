import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ================================
# CONFIG PAGE
# ================================
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="✏️",
    layout="centered"
)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    # Assure-toi que le nom du fichier est correct
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# ================================
# PREPROCESSING (MNIST-COMPATIBLE)
# ================================
def preprocess_image(img):
    """
    Transforme une photo ou un upload en format 28x28 compatible MNIST.
    """
    # --- CONVERSION GRIS ---
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- NETTOYAGE LUMIÈRE (Pour les photos) ---
    # Utilisation d'un seuil adaptatif pour gérer les ombres sur le papier
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # --- ÉPAISSIR LES TRAITS ---
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # --- DÉTECTION DU CHIFFRE ---
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, 0.0

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    quality = min(1.0, cv2.contourArea(cnt) / 500)
    digit = img[y:y+h, x:x+w]

    # --- RESIZE & PADDING ---
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit

    # --- NORMALISATION ---
    padded = padded / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded, quality

# ================================
# TTA PREDICTION
# ================================
def predict_with_tta(img, n=5):
    preds = []
    for _ in range(n):
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        shifted = np.roll(img, shift_x, axis=1)
        shifted = np.roll(shifted, shift_y, axis=2)
        preds.append(model.predict(shifted, verbose=0))
    return np.mean(preds, axis=0)

# ================================
# UI - INTERFACE
# ================================
st.title("📷 Reconnaissance de chiffres")
st.markdown("Prenez une photo d'un chiffre écrit sur papier ou téléchargez un fichier.")

tabs = st.tabs(["📷 Appareil Photo", "📁 Upload Fichier"])

processed = None
quality = 0.0

# --- ONGLET CAMERA ---
with tabs[0]:
    cam_image = st.camera_input("Scanner un chiffre")
    if cam_image:
        img = Image.open(cam_image).convert("RGB")
        processed, quality = preprocess_image(np.array(img))

# --- ONGLET UPLOAD ---
with tabs[1]:
    uploaded = st.file_uploader(
        "Téléverser une image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        processed, quality = preprocess_image(np.array(image))

# ================================
# AFFICHAGE DES RÉSULTATS
# ================================
if processed is not None:
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vision de l'IA")
        st.image(processed.reshape(28, 28), caption="Image après traitement", width=200)

    with col2:
        prediction = predict_with_tta(processed)
        digit = int(np.argmax(prediction))
        confidence = np.max(prediction)

        st.subheader("Résultat")
        st.success(f"✅ Chiffre reconnu : **{digit}**")
        st.metric("Confiance", f"{confidence * 100:.1f} %")

    # Graphique des probabilités
    st.subheader("📊 Probabilités")
    probs = prediction[0]
    for i, p in enumerate(probs):
        st.progress(float(p), text=f"Chiffre {i} : {p*100:.1f}%")

else:
    st.info("💡 Prenez une photo ou téléchargez une image pour lancer l'analyse.")
