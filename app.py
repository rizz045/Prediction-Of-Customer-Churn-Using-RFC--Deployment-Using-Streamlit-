import streamlit as st
import pickle
import numpy as np
import time

# Load the trained model, encoder, and scaler
with open('RFC_Model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('lbl_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below to predict churn.")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.write("This app predicts whether a customer is likely to churn based on their details.")

# Layout Optimization
col1, col2 = st.columns(2)

# Input Fields
with col1:
    state = st.selectbox("State Code", encoder.classes_)
    area_code = st.radio("Area Code", ['415', '408', '510'])
    account_length = st.slider("Account Length", min_value=1, max_value=205, value=100)
    voice_plan = st.radio("Voice Plan", ["No", "Yes"])
    voice_messages = st.slider("Voice Messages", min_value=0, max_value=42, value=10)
    intl_plan = st.radio("International Plan", ["No", "Yes"])
    intl_mins = st.slider("International Minutes", min_value=3.3, max_value=17.2, value=10.0)
    intl_calls = st.slider("International Calls", min_value=1, max_value=10, value=5)

with col2:
    day_mins = st.slider("Day Minutes", min_value=0.0, max_value=351.5, value=180.0)
    day_calls = st.slider("Day Calls", min_value=0, max_value=165, value=80)
    eve_mins = st.slider("Evening Minutes", min_value=0.0, max_value=363.7, value=200.0)
    eve_calls = st.slider("Evening Calls", min_value=0, max_value=170, value=85)
    night_mins = st.slider("Night Minutes", min_value=0.0, max_value=395.0, value=180.0)
    night_calls = st.slider("Night Calls", min_value=0, max_value=175, value=90)
    customer_calls = st.slider("Customer Service Calls", min_value=0, max_value=9, value=2)
    total_charge = st.slider("Gross Total Charges", min_value=31.0, max_value=87.0, value=50.0)

# Encode categorical inputs
state_encoded = encoder.transform([state])[0]
area_code = int(area_code)
voice_plan = 1 if voice_plan == "Yes" else 0
intl_plan = 1 if intl_plan == "Yes" else 0

# Prepare input for model
input_data = np.array([[
    float(state_encoded), float(area_code), float(account_length),
    float(voice_plan), float(voice_messages), float(intl_plan), float(intl_mins), float(intl_calls),
    float(day_mins), float(day_calls), float(eve_mins), float(eve_calls),
    float(night_mins), float(night_calls), float(customer_calls), float(total_charge)
]])

# Scale input
data_scaled = scaler.transform(input_data)

# Predict Button
if st.button("🔍 Predict Churn"):
    with st.spinner("Analyzing customer data... Please wait ⏳"):
        time.sleep(0.1)  # Simulate loading effect
        prediction = model.predict(data_scaled)
        
        # Display Result
        if prediction[0] == 1:
            st.error("🚨 This customer is **likely** to churn! ❌")
        else:
            st.success("✅ This customer is **not likely** to churn!")
