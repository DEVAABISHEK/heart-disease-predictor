import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and scaler (Ensure you ran the pickle.dump cell first!)
try:
    model = pickle.load(open('heart_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the training and pickle cells first.")

st.title("❤️ Cardiovascular Disease Predictor")
st.write("Enter patient metrics to evaluate cardiac risk.")

# Input fields
age = st.slider("Age (Years)", 1, 100, 30)
gender = st.radio("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", value=170)
weight = st.number_input("Weight (kg)", value=70)
ap_hi = st.number_input("Systolic BP (ap_hi)", value=120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", value=80)

if st.button("Analyze Risk"):
    gender_val = 2 if gender == "Male" else 1
    features = np.array([[age*365, gender_val, height, weight, ap_hi, ap_lo, 1, 1, 0, 0, 1]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    
    if prediction[0] == 1:
        st.error("Result: High Risk Identified")
    else:
        st.success("Result: Low Risk Identified")
