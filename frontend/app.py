import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model
classifier = joblib.load('loan_status_model.joblib')

# Define the feature names
feature_names = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
    'Credit_History', 'Property_Area'
]

# Define a function to make predictions
def predict(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = classifier.predict(input_df)
    return prediction[0]

# Streamlit UI
st.title('Loan Status Prediction')

# Input fields
loan_id = st.text_input('Loan ID')
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['No', 'Yes'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['No', 'Yes'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
credit_history = st.selectbox('Credit History', [0, 1])
property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])

# Map input values to model-compatible format
gender = 1 if gender == 'Male' else 0
married = 1 if married == 'Yes' else 0
education = 1 if education == 'Graduate' else 0
self_employed = 1 if self_employed == 'Yes' else 0
property_area = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}[property_area]
dependents = 4 if dependents == '3+' else int(dependents)

input_data = [gender, married, dependents, education, self_employed, applicant_income,
              coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]

if st.button('Predict'):
    result = predict(input_data)
    print(result)
    if result == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Denied')
