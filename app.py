import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ================================
# CONFIGURATION DE LA PAGE
# ================================
st.set_page_config(
    page_title="Reconnaissance MNIST Pro",
    page_icon="🔢",
    layout="centered"
)

# ================================
# CHARGEMENT DU MODÈLE
# ================================
@st.cache_resource
def load_model():
    # Remplace "mnist_model.keras" par le nom exact de ton fichier
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# ================================
# PRÉTRAITEMENT ROBUSTE (PHOTO & FICHIER)
# ================================
def preprocess_digit(img_array):
    """
    Transforme une image (caméra ou upload) en format MNIST 28x28.
    Gère les traits fins et les variations de lumière.
    """
    # 1. Conversion en niveaux de gris
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 2. Amélioration du contraste pour les écritures fines (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. Flou pour réduire le grain de la photo
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Binarisation Adaptative (indispensable pour les photos réelles)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 5. Détection du contenu pour recadrage
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # On prend le plus grand contour (le chiffre)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    roi = thresh[y:y+h, x:x+w]

    # 6. Épaississement dynamique (Dilation) si le trait est trop fin
    kernel = np.ones((3,3), np.uint8)
    if cv2.countNonZero(roi) < (roi.shape[0] * roi.shape[1] * 0.2):
        roi = cv2.dilate(roi, kernel, iterations=1)

    # 7. Redimensionnement vers 20x20 (en gardant les proportions)
    final_size = 20
    mask = np.zeros((final_size, final_size), dtype=np.uint8)
    if w > h:
        new_w = final_size
        new_h = int(h * (final_size / w))
    else:
        new_h = final_size
        new_w = int(w * (final_size / h))
    
    roi_res = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Centrage dans le masque 20x20
    xx = (final_size - new_w) // 2
    yy = (final_size - new_h) // 2
    mask[yy:yy+new_h, xx:xx+new_w] = roi_res

    # 8. Padding final pour arriver à 28x28
    padded = cv2.copyMakeBorder(mask, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    # 9. Normalisation pour le modèle
    final_img = padded.astype('float32') / 255.0
    return final_img.reshape(1, 28, 28, 1)

# ================================
# INTERFACE UTILISATEUR (UI)
# ================================
st.title("🔢 Reconnaissance de Chiffres")
st.write("Choisissez votre méthode pour soumettre un chiffre manuscrit (0-9).")

# Création des onglets
tab_camera, tab_upload = st.tabs(["📷 Appareil Photo", "📁 Importer une Image"])

img_source = None

with tab_camera:
    camera_input = st.camera_input("Prendre une photo du chiffre")
    if camera_input:
        img_source = camera_input

with tab_upload:
    file_input = st.file_uploader("Choisir une image sur mon PC", type=['png', 'jpg', 'jpeg'])
    if file_input:
        img_source = file_input

# ================================
# TRAITEMENT ET PRÉDICTION
# ================================
if img_source is not None:
    # Conversion de l'entrée en tableau Numpy
    image = Image.open(img_source).convert("RGB")
    image_np = np.array(image)

    processed_img = preprocess_digit(image_np)

    if processed_img is not None:
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Aperçu IA")
            # On affiche l'image traitée en 28x28 (agrandie pour la visibilité)
            st.image(processed_img.reshape(28, 28), caption="Image normalisée", width=200)

        with col2:
            # Exécution de la prédiction
            prediction = model.predict(processed_img, verbose=0)
            digit = np.argmax(prediction)
            prob = np.max(prediction)

            st.subheader("Résultat")
            if prob > 0.8:
                st.success(f"Chiffre détecté : **{digit}**")
            else:
                st.warning(f"Chiffre probable : **{digit}**")
            
            st.metric("Confiance", f"{prob*100:.1f} %")

        # Détails des probabilités
        with st.expander("Voir les probabilités détaillées"):
            for i, val in enumerate(prediction[0]):
                st.write(f"Chiffre {i} : {val*100:.2f}%")
                st.progress(float(val))
    else:
        st.error("Aucun chiffre n'a été détecté sur l'image. Assurez-vous que le contraste est suffisant.")

else:
    st.info("Veuillez prendre une photo ou télécharger un fichier pour lancer l'analyse.")
