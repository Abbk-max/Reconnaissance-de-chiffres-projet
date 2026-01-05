import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ================================
# CONFIGURATION DE LA PAGE
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
    # Remplace par le nom exact de ton fichier .keras
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# ================================
# PRÉTRAITEMENT OPTIMISÉ PHOTO
# ================================
def preprocess_image(img_array):
    """
    Nettoie et convertit une photo réelle au format MNIST (28x28).
    Gère les ombres et les traits fins.
    """
    # 1. Conversion en niveaux de gris
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 2. Amélioration du contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. Réduction du bruit et seuillage adaptatif
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Gère les variations de lumière sur le papier
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 4. Épaississement des traits (Dilation)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 5. Détection et extraction du chiffre
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # On prend le contour le plus imposant
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    roi = thresh[y:y+h, x:x+w]

    # 6. Redimensionnement 20x20 et padding pour arriver à 28x28
    roi_res = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(roi_res, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    # 7. Normalisation
    return padded.astype('float32') / 255.0

# ================================
# INTERFACE UTILISATEUR
# ================================
st.title("🔢 Scanner de Chiffres intelligent")
st.write("Soumettez un chiffre via votre caméra ou un fichier local.")

# Sélection de la source
option = st.radio("Source de l'image :", ["📷 Caméra en direct", "📁 Télécharger un fichier"], horizontal=True)

img_buffer = None

if option == "📷 Caméra en direct":
    img_buffer = st.camera_input("Prendre une photo")
else:
    img_buffer = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])

# ================================
# ANALYSE ET FILTRE DE CONFIANCE
# ================================
if img_buffer:
    # Lecture de l'image
    image = Image.open(img_buffer).convert("RGB")
    image_np = np.array(image)
    
    # Prétraitement
    processed = preprocess_image(image_np)

    if processed is not None:
        # Préparation pour le modèle
        input_data = processed.reshape(1, 28, 28, 1)
        
        # Prédiction
        prediction = model.predict(input_data, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ce que l'IA voit")
            st.image(processed, caption="Image normalisée (28x28)", width=150)

        with col2:
            st.subheader("Analyse")
            
            # --- LOGIQUE D'INTERVALLE DE CONFIANCE ---
            if confidence >= 0.80:
                st.success(f"**Chiffre reconnu : {digit}**")
                st.balloons() # Célébration si confiance élevée
            elif confidence >= 0.50:
                st.warning(f"**Chiffre probable : {digit}**")
                st.info("La qualité de l'image est moyenne.")
            else:
                st.error("L'IA n'est pas sûre d'elle. Essayez de mieux cadrer le chiffre.")

            st.metric("Niveau de confiance", f"{confidence*100:.1f} %")

        # Détail des probabilités
        with st.expander("📊 Voir le détail des probabilités"):
            for i, p in enumerate(prediction[0]):
                st.write(f"Chiffre {i} : {p*100:.1f}%")
                st.progress(float(p))
    else:
        st.error("Impossible de détecter un tracé. Écrivez plus gros ou vérifiez l'éclairage.")

else:
    st.info("En attente d'une image pour analyse.")
