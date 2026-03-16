import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Load the saved model and scaler
try:
    model = pickle.load(open('heart_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'heart_model.pkl' or 'scaler.pkl' not found in repository.")

st.title("❤️ Cardiovascular Disease Predictor")
st.write("Enter patient metrics below to evaluate potential cardiac risk.")

# Input fields arranged in columns for a better UI
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 1, 100, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", value=170)

with col2:
    weight = st.number_input("Weight (kg)", value=70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", value=80)

# Add remaining features with default values (chol, gluc, smoke, alco, active)
st.info("Note: Prediction uses average values for lifestyle factors (Smoking, Alcohol, Activity).")

if st.button("Analyze Risk"):
    gender_val = 2 if gender == "Male" else 1
    # Standardizing feature vector to 11 inputs: 
    # age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    features = np.array([[age*365, gender_val, height, weight, ap_hi, ap_lo, 1, 1, 0, 0, 1]])
    
    # Pre-processing
    scaled = scaler.transform(features)
    
    # Predicting
    prediction = model.predict(scaled)
    
    st.subheader("Final Result:")
    if prediction[0] == 1:
        st.error("⚠️ High Risk Identified: Please consult a medical professional.")
    else:
        st.success("✅ Low Risk Identified: Continue maintaining a healthy lifestyle!")
