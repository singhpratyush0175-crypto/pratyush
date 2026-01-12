import streamlit as st
import joblib
import numpy as np

st.title("Student Pass/Fail Prediction System")

model = joblib.load("student_pass_model.pkl")

study_hours = st.number_input(
    "Study Hours per Week", 0.0, 40.0, 10.0
)

attendance = st.number_input(
    "Attendance Percentage", 0.0, 100.0, 75.0
)

if st.button("Predict Result"):
    input_data = np.array([[study_hours, attendance]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Prediction: PASS")
    else:
        st.error("Prediction: FAIL")
