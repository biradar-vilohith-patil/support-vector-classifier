import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ad Conversion AI", layout="wide")

# Load Custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load Artifacts & Data
@st.cache_resource
def load_assets():
    with open('models/svc_model.pkl', 'rb') as m, open('models/svc_scaler.pkl', 'rb') as s:
        model, scaler = pickle.load(m), pickle.load(s)
    data = pd.read_csv('data/Social_Network_Ads.csv')
    return model, scaler, data

model, scaler, df = load_assets()

st.title("🎯 Customer Conversion Predictor")
st.markdown("Adjust the demographic profiles below to see if a customer is likely to purchase the product.")

# Create a sleek two-column layout
col_inputs, col_graph = st.columns([1, 2], gap="large")

with col_inputs:
    st.markdown("### 🎛️ User Profile")
    age = st.slider("Customer Age", min_value=18, max_value=60, value=30, step=1)
    salary = st.slider("Estimated Salary ($)", min_value=15000, max_value=150000, value=50000, step=1000)
    
    # Prediction Trigger
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Conversion"):
        features = scaler.transform(np.array([[age, salary]]))
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][prediction] * 100
        
        if prediction == 1:
            st.success(f"### 🛒 Likely to Buy\nConfidence: **{prob:.1f}%**")
        else:
            st.error(f"### ❌ Unlikely to Buy\nConfidence: **{prob:.1f}%**")
    else:
        # Default state
        prediction = None

with col_graph:
    st.markdown("### 📊 Market Positioning")
    # Base scatter plot of historical data
    fig = px.scatter(df, x='Age', y='EstimatedSalary', color='Purchased',
                     color_continuous_scale=['#ff4b4b', '#00b894'],
                     labels={'Purchased': 'Bought (1=Yes, 0=No)'},
                     opacity=0.6)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Overlay the user's current input as a large marker
    fig.add_trace(go.Scatter(
        x=[age], y=[salary],
        mode='markers',
        marker=dict(size=20, color='#feca57', line=dict(width=3, color='black'), symbol='star'),
        name='Current Target'
    ))
    
    st.plotly_chart(fig, use_container_width=True)