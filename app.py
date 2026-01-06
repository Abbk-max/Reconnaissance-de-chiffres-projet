import streamlit as st
import numpy as np
import random
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Import backend logic
import mnist_core as backend

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="✏️",
    layout="centered"
)

# Load resources once
model = backend.load_trained_model()
x_test, y_test = backend.load_mnist_test_data()

# ==============================================================================
# 2. UI COMPONENTS (FUNCTIONS)
# ==============================================================================

def render_header():
    st.title("📷 Reconnaissance de chiffres manuscrits")
    st.caption("Développé par **ABOUBAKAR Ramzy**")
    st.markdown("---")

def render_results(processed_img, quality):
    """Displays the analysis results section."""
    if processed_img is None:
        return

    st.divider()
    st.subheader("🧠 Analyse")

    cols = st.columns([1, 2])
    
    # Left Column: Input Visualization
    with cols[0]:
        st.image(
            processed_img.reshape(28, 28),
            caption="Vue Modèle (28x28)",
            width=100,
            clamp=True
        )
        st.metric("Qualité Détection", f"{quality*100:.0f}%")

    # Right Column: Prediction
    with cols[1]:
        prediction = backend.predict_digit(model, processed_img)
        digit = int(np.argmax(prediction))
        confidence = np.max(prediction)

        if confidence > 0.8:
            st.success(f"## Chiffre : {digit}")
            st.caption(f"Confiance : {confidence*100:.2f}%")
        else:
            st.warning(f"## Chiffre : {digit} ?")
            st.caption(f"Confiance : {confidence*100:.2f}% (Incertain)")

    # Probabilities Chart
    st.bar_chart(prediction[0])


# ==============================================================================
# 3. MODE HANDLERS
# ==============================================================================

def handle_camera_mode():
    camera_image = st.camera_input("Prendre une photo")
    if camera_image:
        img_pil = Image.open(camera_image).convert("RGB")
        return backend.process_image_pipeline(img_pil)
    return None, 0.0

def handle_upload_mode():
    uploaded = st.file_uploader("Image (png, jpg)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        return backend.process_image_pipeline(img_pil)
    return None, 0.0

def handle_drawing_mode():
    st.write("Dessine un chiffre en **blanc sur fond noir**.")
    
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        key="canvas",
    )
    
    # State handling for the drawing validation
    if canvas_result.image_data is not None:
        if st.button("Valider le dessin"):
            img_data = canvas_result.image_data.astype('uint8')
            if np.sum(img_data) > 0:
                img_pil = Image.fromarray(img_data).convert("RGB")
                processed, quality = backend.process_image_pipeline(img_pil)
                st.session_state['drawing_processed'] = processed
                return processed, quality
        
        elif 'drawing_processed' in st.session_state:
             return st.session_state['drawing_processed'], 1.0
             
    return None, 0.0

def handle_mnist_mode():
    if st.button("🎲 Choisir un chiffre aléatoire"):
        idx = random.randint(0, len(x_test) - 1)
        img_mnist = x_test[idx]
        label_mnist = y_test[idx]
        
        # Display Ground Truth
        st.image(img_mnist, caption=f"Vérité terrain : {label_mnist}", width=150)
            
        # Prepare for model
        img_norm = img_mnist.astype('float32') / 255.0
        processed = img_norm.reshape(1, 28, 28, 1)
        
        st.session_state['random_mnist_processed'] = processed
        return processed, 1.0
    
    elif 'random_mnist_processed' in st.session_state:
        # Re-display logic could be added here if needed, 
        # but for now we just return the state
        return st.session_state['random_mnist_processed'], 1.0
        
    return None, 0.0

# ==============================================================================
# 4. MAIN APP LOOP
# ==============================================================================

def main():
    render_header()

    # Mode Selector
    modes = {
        "📷 Caméra": handle_camera_mode,
        "📁 Upload": handle_upload_mode,
        "✏️ Dessiner": handle_drawing_mode,
        "🎲 MNIST": handle_mnist_mode
    }
    
    selected_mode_name = st.radio("Mode :", list(modes.keys()), horizontal=True)

    # State Reset on Mode Change
    if 'last_mode' not in st.session_state:
        st.session_state['last_mode'] = selected_mode_name

    if st.session_state['last_mode'] != selected_mode_name:
        # Clear specific session keys
        keys_to_clear = ['drawing_processed', 'random_mnist_processed']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['last_mode'] = selected_mode_name
        st.rerun()

    # Execute selected mode handler
    handler_func = modes[selected_mode_name]
    processed_img, quality = handler_func()

    # Show results
    render_results(processed_img, quality)

if __name__ == "__main__":
    main()