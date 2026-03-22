import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

# Set page config for a wider layout
st.set_page_config(page_title="Malaria Cell Analyzer", layout="wide")

st.title("🔬 Malaria Parasite Detection System")
st.write("Upload a blood smear cell image to see the algorithm's thought process and final prediction.")

# ==========================================
# 1. HELPER FUNCTIONS (Adapted for arrays)
# ==========================================
def process_uploaded_image(uploaded_file):
    # Convert Streamlit uploaded file (bytes) to OpenCV image (BGR array)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

def extract_features_from_array(img_bgr, size=(128, 128)):
    # This is your exact mathematical feature extractor, adapted to take an array
    img_bgr = cv2.resize(img_bgr, size)
    b, g, r = cv2.split(img_bgr)
    m_b, m_g, m_r = np.mean(b), np.mean(g), np.mean(r)
    m = (m_b + m_g + m_r) / 3.0

    if m_b != 0 and m_g != 0 and m_r != 0:
        b_norm = np.clip(b * (m / m_b), 0, 255).astype(np.uint8)
        g_norm = np.clip(g * (m / m_g), 0, 255).astype(np.uint8)
        r_norm = np.clip(r * (m / m_r), 0, 255).astype(np.uint8)
        img_norm = cv2.merge((b_norm, g_norm, r_norm))
    else:
        img_norm = img_bgr.copy()
        g_norm = g

    gray_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    _, cell_mask = cv2.threshold(gray_norm, 10, 255, cv2.THRESH_BINARY)
    
    local_thresh = cv2.adaptiveThreshold(g_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    masked_thresh = cv2.bitwise_and(local_thresh, cell_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    actual_spots = cv2.morphologyEx(masked_thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(actual_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_spots = len(contours)
    
    if num_spots > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_spot_area = max(areas)
        total_spot_area = sum(areas)
        
        img_hsv = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
        saturation_channel = img_hsv[:, :, 1]
        
        c_max = contours[np.argmax(areas)]
        max_spot_mask = np.zeros_like(gray_norm)
        cv2.drawContours(max_spot_mask, [c_max], -1, 255, -1)
        spot_saturation = np.mean(saturation_channel[max_spot_mask == 255])
    else:
        max_spot_area, total_spot_area, spot_saturation = 0, 0, 0

    return np.array([num_spots, max_spot_area, total_spot_area, spot_saturation, np.std(gray_norm[cell_mask == 255])])

def generate_visuals(img_bgr, size=(128, 128)):
    # Your exact visualizer, returning a Matplotlib figure instead of plt.show()
    img_bgr = cv2.resize(img_bgr, size)
    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    b, g, r = cv2.split(img_bgr)
    m_b, m_g, m_r = np.mean(b), np.mean(g), np.mean(r)
    m = (m_b + m_g + m_r) / 3.0

    if m_b != 0 and m_g != 0 and m_r != 0:
        b_norm = np.clip(b * (m / m_b), 0, 255).astype(np.uint8)
        g_norm = np.clip(g * (m / m_g), 0, 255).astype(np.uint8)
        r_norm = np.clip(r * (m / m_r), 0, 255).astype(np.uint8)
        img_norm = cv2.merge((b_norm, g_norm, r_norm))
    else:
        img_norm = img_bgr.copy()
        g_norm = g

    img_rgb_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
    gray_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    _, cell_mask = cv2.threshold(gray_norm, 10, 255, cv2.THRESH_BINARY)
    
    local_thresh = cv2.adaptiveThreshold(g_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    masked_thresh = cv2.bitwise_and(local_thresh, cell_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    actual_spots = cv2.morphologyEx(masked_thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(actual_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = img_rgb_norm.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.patch.set_facecolor('none') # Make background transparent for Streamlit
    
    axes[0, 0].imshow(img_rgb_original); axes[0, 0].set_title("1. Original Image"); axes[0, 0].axis('off')
    axes[0, 1].imshow(img_rgb_norm); axes[0, 1].set_title("2. Color Normalized"); axes[0, 1].axis('off')
    axes[0, 2].imshow(g_norm, cmap='gray'); axes[0, 2].set_title("3. Green Channel"); axes[0, 2].axis('off')
    axes[1, 0].imshow(local_thresh, cmap='gray'); axes[1, 0].set_title("4. Raw Adaptive Thresh"); axes[1, 0].axis('off')
    axes[1, 1].imshow(actual_spots, cmap='gray'); axes[1, 1].set_title("5. Cleaned Spots"); axes[1, 1].axis('off')
    axes[1, 2].imshow(img_with_contours); axes[1, 2].set_title(f"6. Final Detections ({len(contours)} spots)"); axes[1, 2].axis('off')

    plt.tight_layout()
    return fig

# ==========================================
# 2. STREAMLIT UI & LOGIC
# ==========================================

# NOTE: You will need to save your best optimized pipeline using joblib in your training script:
# import joblib
# joblib.dump(best_pipeline, 'best_malaria_model.pkl')

@st.cache_resource # Caches the model so it doesn't reload on every button click
def load_model():
    try:
        return joblib.load('./best_model.pkl')
    except:
        return None

model = load_model()

uploaded_file = st.file_uploader("Upload a cell image (.png or .jpg)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 1. Process the image
    img_bgr = process_uploaded_image(uploaded_file)
    
    # 2. Extract Features
    features = extract_features_from_array(img_bgr)
    
    st.markdown("---")
    st.subheader("Algorithm Step-by-Step Visualization")
    
    # 3. Generate and display the plot
    with st.spinner('Generating visual analysis...'):
        fig = generate_visuals(img_bgr)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Diagnostic Prediction")
    
    # 4. Make Prediction
    if model is None:
        st.warning("⚠️ Could not find 'best_malaria_model.pkl'. Please ensure your trained model is saved in the same directory.")
    else:
        # The model expects a 2D array, so we reshape our 1D feature array
        features_2d = features.reshape(1, -1)
        prediction = model.predict(features_2d)[0]
        
        # Display results with nice formatting
        if prediction == 0: # Assuming 1 is Parasitized
            st.error("🚨 **PREDICTION: PARASITIZED**")
            st.write(f"The algorithm detected **{int(features[0])}** distinct parasitic spot(s).")
        else: # Assuming 0 is Healthy
            st.success("✅ **PREDICTION: UNINFECTED (HEALTHY)**")
            st.write("The algorithm did not detect any significant parasitic spots after normalization.")