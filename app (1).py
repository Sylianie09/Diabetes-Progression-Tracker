import streamlit as st
import joblib
import numpy as np

model = joblib.load('diabetes_progression_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Diabetes Progression Tracker')
st.write('Enter values below to get a prediction.')

# Example input — repeat st.number_input for each feature in your dataset
age = st.number_input('Age')
bmi = st.number_input('Body Mass Index(BMI)')
s1  = st.number_input('Total Cholesterol')
s3  = st.number_input('High Density Lipoprotein(HDL)')
s4  = st.number_input('Cholesterol/HDL Ratio')
s6  = st.number_input('Glucose Level')

if st.button('Predict'):
    input_data = np.array([[age, bmi, s1, s3, s4, s6]])
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
    Prediction = float(result[0])

# Adding visuals to emphasize risk
    if prediction < 100:
        st.success(f"Prediction Score: {prediction:.2f} — Low Disease Progression Risk")
    elif 100 <= prediction < 200:
        st.warning(f"Prediction Score: {prediction:.2f} — Moderate Disease Progression Risk")
    else:
        st.error(f"Prediction Score: {prediction:.2f} — High Disease Progression Risk!")

st.caption("This is a learning demo, not a real medical tool.")
