import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Mobile Tier Predictor", page_icon="📱")

# Inject custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    with open('svc_model.pkl', 'rb') as m, open('svc_scaler.pkl', 'rb') as s:
        return pickle.load(m), pickle.load(s)

model, scaler = load_artifacts()

# UI Headers
st.title("📱 Smart Device Tier Predictor")
st.markdown("Adjust the hardware specifications below to predict the retail price tier.")

# Simple Slider Inputs
ram = st.slider("RAM Capacity (MB)", min_value=256, max_value=4096, value=2048, step=128)
battery = st.slider("Battery Power (mAh)", min_value=500, max_value=2000, value=1200, step=50)
px_width = st.slider("Screen Width (Pixels)", min_value=500, max_value=2000, value=1080, step=10)
px_height = st.slider("Screen Height (Pixels)", min_value=0, max_value=2000, value=1920, step=10)

# Prediction Logic
if st.button("Predict Price Tier"):
    # Format and scale the input exactly as done in the notebook
    input_features = np.array([[ram, battery, px_width, px_height]])
    scaled_features = scaler.transform(input_features)
    
    # Run prediction
    prediction = model.predict(scaled_features)[0]
    
    # Map the numeric output to a human-readable label
    tier_mapping = {
        0: ("Budget Tier", "🪙"),
        1: ("Mid-Range Tier", "💵"),
        2: ("Premium Tier", "💳"),
        3: ("Flagship Tier", "💎")
    }
    
    tier_name, icon = tier_mapping[prediction]
    
    # Display results
    st.success(f"### {icon} Predicted Segment: **{tier_name}**")