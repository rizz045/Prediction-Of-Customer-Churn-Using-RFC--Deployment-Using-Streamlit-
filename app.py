import streamlit as st
import pickle
import numpy as np
import time

# Load Model, Encoder & Scaler
with open('RFC_Model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('lbl_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.image("https://thumbs.dreamstime.com/b/customer-churn-mark-charts-inscription-293698142.jpg", use_container_width=True)

# Custom Styling
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction</h1>
    <p style='text-align: center;'>Enter customer details to predict churn.</p>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("Enter customer details below to predict churn.")

# Input Section
st.markdown("---")
st.subheader("🔹 Customer Details")

# Input Fields (Better UI with Cards)
with st.container():
    state = st.selectbox("🌍 State Code", 
                         ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY',
                          'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA',
                          'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM',
                          'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'])

    state_encoded = encoder.transform([state])

    area_code = st.selectbox("📞 Area Code", ['415', '408', '510'])
    account_length = st.number_input("📆 Account Length (Days)", min_value=1, max_value=205, step=1)

    voice_plan = st.radio("📢 Voice Plan", ["No", "Yes"])
    voice_messages = st.slider("📩 Voice Messages", min_value=0, max_value=42, step=1)

    intl_plan = st.radio("🌎 International Plan", ["No", "Yes"])
    intl_mins = st.number_input("⏳ International Minutes", min_value=3.3, max_value=17.2, step=0.1, format="%.2f")
    intl_calls = st.slider("📞 International Calls", min_value=1, max_value=10, step=1)

    day_mins = st.number_input("🌞 Day Minutes", min_value=0.0, max_value=351.5, step=1.0, format="%.2f")
    day_calls = st.slider("📲 Day Calls", min_value=0, max_value=165, step=1)

    eve_mins = st.number_input("🌙 Evening Minutes", min_value=0.0, max_value=363.7, step=1.0, format="%.2f")
    eve_calls = st.slider("📞 Evening Calls", min_value=0, max_value=170, step=1)

    night_mins = st.number_input("🌜 Night Minutes", min_value=0.0, max_value=395.0, step=1.0, format="%.2f")
    night_calls = st.slider("📱 Night Calls", min_value=0, max_value=175, step=1)

    customer_calls = st.slider("📞 Customer Service Calls", min_value=0, max_value=9, step=1)
    total_charge = st.number_input("💰 Gross Total Charges", min_value=31.0, max_value=87.0, step=0.1, format="%.2f")

# Convert Inputs
area_code = int(area_code)
voice_plan = 1 if voice_plan == "Yes" else 0
intl_plan = 1 if intl_plan == "Yes" else 0

# Prepare Data for Prediction
input_data = np.array([[float(state_encoded[0]), float(area_code), float(account_length),
                        float(voice_plan), float(voice_messages), float(intl_plan), float(intl_mins), float(intl_calls),
                        float(day_mins), float(day_calls), float(eve_mins), float(eve_calls),
                        float(night_mins), float(night_calls), float(customer_calls), float(total_charge)]])
input_data_scaled = scaler.transform(input_data)

# Predict Button with Loading Animation
if st.button("🔍 Predict Churn"):
    with st.spinner("Analyzing customer data..."):
        time.sleep(1)  # Short delay for loading animation
        prediction = model.predict(input_data_scaled)

    # Display Result with Icon
    if prediction[0] == 1:
        st.error("❌ This customer is **likely** to churn.")
    else:
        st.success("✅ This customer is **not likely** to churn.")

# Footer
st.markdown("---")
st.markdown("<h5 style='text-align:center;'>Developed with ❤️ using Streamlit</h5>", unsafe_allow_html=True)
