import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ================================
# CONFIG PAGE
# ================================
st.set_page_config(
    page_title="Scanner de Chiffres AI",
    page_icon="📷",
    layout="centered"
)

# ================================
# CHARGEMENT DU MODÈLE
# ================================
@st.cache_resource
def load_model():
    # Assure-toi que le nom du fichier correspond au tien
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# ================================
# PRÉTRAITEMENT OPTIMISÉ PHOTO
# ================================
def preprocess_image(img):
    """
    Transforme une photo réelle en format MNIST (28x28, noir et blanc, centré)
    """
    # 1. Conversion en Gris
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Amélioration du contraste (utile pour les photos avec ombre)
    img = cv2.equalizeHist(img)

    # 3. Flou pour réduire le bruit de la photo
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 4. Binarisation Adaptative (mieux que le seuil fixe pour les photos)
    # Elle gère les différences d'éclairage sur la feuille
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 5. Nettoyage des petits points (bruit)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # 6. Épaississement dynamique pour les écritures fines
    img = cv2.dilate(img, kernel, iterations=1)

    # 7. Détection et Recadrage
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # On extrait le chiffre et on lui donne une marge
    digit = img[y:y+h, x:x+w]
    
    # Redimensionnement vers 20x20 (format standard MNIST avant padding)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # 8. Mise au format 28x28 centré
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit

    # 9. Normalisation
    final_img = padded.astype('float32') / 255.0
    final_img = final_img.reshape(1, 28, 28, 1)
    
    quality = min(1.0, cv2.contourArea(cnt) / 500)
    return final_img, quality

# ================================
# UI : INTERFACE UTILISATEUR
# ================================
st.title("📷 Scanner de Chiffres Manuscrits")
st.write("Prenez une photo d'un chiffre (0-9) écrit sur papier.")

# Utilisation de l'appareil photo
camera_image = st.camera_input("Scanner un chiffre")

processed = None
quality = 0.0

if camera_image:
    # Conversion de l'image de la caméra en format OpenCV
    img_pil = Image.open(camera_image)
    img_array = np.array(img_pil)
    
    # Prétraitement
    processed, quality = preprocess_image(img_array)

# ================================
# PRÉDICTION ET RÉSULTATS
# ================================
if processed is not None:
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vision de l'IA")
        st.image(processed.reshape(28, 28), caption="Image après traitement", width=200)

    with col2:
        # Prédiction simple (ou TTA si tu veux garder ta fonction précédente)
        prediction = model.predict(processed, verbose=0)
        digit_final = np.argmax(prediction)
        confidence = np.max(prediction)

        st.subheader("Analyse")
        if confidence > 0.7:
            st.success(f"Chiffre détecté : **{digit_final}**")
        else:
            st.warning(f"Chiffre probable : **{digit_final}** (Confiance faible)")
        
        st.metric("Fiabilité", f"{confidence*100:.1f}%")

    # Détails des probabilités
    with st.expander("Voir les détails par chiffre"):
        probs = prediction[0]
        for i, p in enumerate(probs):
            st.progress(float(p), text=f"Chiffre {i} : {p*100:.1f}%")

else:
    if camera_image:
        st.error("Impossible de détecter un chiffre. Essayez d'écrire plus gros ou de mieux éclairer la feuille.")
    else:
        st.info("Autorisez l'accès à votre caméra pour commencer.")
