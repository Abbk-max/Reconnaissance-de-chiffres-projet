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
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# ================================
# PREPROCESSING (MNIST-COMPATIBLE)
# ================================
def preprocess_image(img):
    """
    Input:
        img : np.array (RGB ou RGBA)
    Output:
        image 28x28 normalisée (1,28,28,1)
        quality score [0,1]
    """

    # --- RGBA -> RGB ---
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # --- RGB -> GRAYSCALE ---
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- Blur léger ---
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # --- Binarisation MNIST ---
    # Pour une photo, OTSU est très utile pour séparer le chiffre du fond
    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- Épaissir les traits ---
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # --- Détection du chiffre ---
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, 0.0

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    quality = min(1.0, cv2.contourArea(cnt) / 500)

    digit = img[y:y+h, x:x+w]

    # --- Resize MNIST ---
    digit = cv2.resize(digit, (20, 20))

    # --- Padding centré ---
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit

    # --- Normalisation ---
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
# UI
# ================================
st.title("📷 Reconnaissance de chiffres manuscrits")
st.markdown("Prends une photo d'un chiffre **(0–9)** ou téléverse une image.")

tabs = st.tabs(["📷 Appareil Photo", "📁 Upload"])

processed = None
quality = 0.0

# ================================
# CAMERA TAB (Remplaçant le Dessin)
# ================================
with tabs[0]:
    camera_image = st.camera_input("Prendre une photo du chiffre")

    if camera_image is not None:
        img_pil = Image.open(camera_image).convert("RGB")
        img_array = np.array(img_pil)
        processed, quality = preprocess_image(img_array)

# ================================
# UPLOAD TAB
# ================================
with tabs[1]:
    uploaded = st.file_uploader(
        "Téléverser une image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img = np.array(image)
        processed, quality = preprocess_image(img)

# ================================
# PREDICTION
# ================================
if processed is not None:
    st.subheader("🧠 Résultat")

    # --- Debug visuel ---
    st.image(
        processed.reshape(28, 28),
        caption="Image normalisée envoyée au modèle (28×28)",
        clamp=True,
        width=150
    )

    prediction = predict_with_tta(processed)
    digit = int(np.argmax(prediction))
    confidence = np.max(prediction)

    # Affichage avec intervalle de confiance
    if confidence > 0.8:
        st.success(f"✅ Chiffre reconnu : **{digit}** ({confidence*100:.1f}%)")
    else:
        st.warning(f"🤔 Chiffre probable : **{digit}** ({confidence*100:.1f}%)")

    st.metric(
        "Score qualité du prétraitement",
        f"{quality * 100:.1f} %"
    )

    st.subheader("📊 Probabilités")
    probs = prediction[0]

    for i, p in enumerate(probs):
        st.progress(float(p), text=f"Chiffre {i} : {p*100:.1f}%")

else:
    st.info("📷 Prends une photo ou téléverse un chiffre pour lancer la reconnaissance.")
