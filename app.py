import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("intervention_model.pkl")

st.title("Meningioma Intervention Risk Calculator")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=65)
sex = st.selectbox("Sex", options=["Male", "Female"])
antal = st.slider("Number of meningiomas", min_value=1, max_value=5, value=1)
edema = st.selectbox("Peritumoral edema", options=["No", "Yes"])
calc = st.selectbox("Calcification", options=["No", "Yes"])
location = st.selectbox("Location", options=["Infratentorial", "Supratentorial"])
width = st.number_input("Tumor width (cm)", value=20.0)
length = st.number_input("Tumor length (cm)", value=25.0)
epilepsy = st.selectbox("Epilepsy", options=["No", "Yes"])
pressure_symptoms = st.selectbox("Pressure symptoms", options=["No", "Yes"])
focal_symptoms = st.selectbox("Focal symptoms", options=["No", "Yes"])

# Convert to model input format
input_data = np.array([[
    age,
    1 if sex == "Female" else 0,
    antal,
    1 if edema == "Yes" else 0,
    1 if calc == "Yes" else 0,
    1 if location == "Supratentorial" else 0,
    width,
    length,
    1 if epilepsy == "Yes" else 0,
    1 if pressure_symptoms == "Yes" else 0,
    1 if focal_symptoms == "Yes" else 0
]])

# Predict
if st.button("Predict Risk"):
    probability = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted 10-year intervention risk: {probability:.2%}")
