# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model, encoder, and scaler
# with open('RFC_Model.pkl', 'rb') as file:
#     model = pickle.load(file)
# with open('lbl_encoder.pkl', 'rb') as file:
#     encoder = pickle.load(file)
# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# # Streamlit UI
# st.title("Customer Churn Prediction")
# st.write("Enter customer details to predict churn.")

# # Input Fields
# state = st.selectbox("State Code", ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY',
#  'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA',
#  'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM',
#  'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'])

# # Fixing Label Encoding Issue
# state_encoded = encoder.transform([state])  # Wrap in a list

# area_code = st.selectbox("Area Code", ['415', '408', '510'])
# account_length = st.number_input("Account Length", min_value=1, max_value=205, step=1)

# voice_plan = st.selectbox("Voice Plan", ["no", "yes"])
# voice_messages = st.number_input("Voice Messages", min_value=0, max_value=42, step=1)

# intl_plan = st.selectbox("International Plan", ["no", "yes"])
# intl_mins = st.number_input("International Minutes", min_value=3.3, max_value=17.2, step=0.1, format="%.2f")
# intl_calls = st.number_input("International Calls", min_value=1, max_value=10, step=1)

# day_mins = st.number_input("Day Minutes", min_value=0.0, max_value=351.5, step=1.0, format="%.2f")
# day_calls = st.number_input("Day Calls", min_value=0, max_value=165, step=1)

# eve_mins = st.number_input("Evening Minutes", min_value=0.0, max_value=363.7, step=1.0, format="%.2f")
# eve_calls = st.number_input("Evening Calls", min_value=0, max_value=170, step=1)

# night_mins = st.number_input("Night Minutes", min_value=0.0, max_value=395.0, step=1.0, format="%.2f")
# night_calls = st.number_input("Night Calls", min_value=0, max_value=175, step=1)

# customer_calls = st.number_input("Customer Service Calls", min_value=0, max_value=9, step=1)

# # total_charge = st.number_input("Gross Total Charges", min_value=31, max_value=87, step=1, format="%.2f")
# total_charge = st.number_input("Gross Total Charges", min_value=31.0, max_value=87.0, step=0.1, format="%.2f")

# # Fixing Proper Format for Input
# area_code = int(area_code)

# voice_plan = 1 if voice_plan == "yes" else 0
# intl_plan = 1 if intl_plan == "yes" else 0

# # Fixing Duplicate Input Data Conversion
# input_data = np.array([[float(state_encoded[0]), float(area_code), float(account_length),
#                         float(voice_plan), float(voice_messages), float(intl_plan), float(intl_mins), float(intl_calls),
#                         float(day_mins), float(day_calls), float(eve_mins), float(eve_calls),
#                         float(night_mins), float(night_calls), float(customer_calls), float(total_charge)]])

# # Ensure Correct Shape Before Scaling
# input_data_scaled = scaler.transform(input_data)

# # Predict Button
# if st.button("Predict Churn"):
#     prediction = model.predict(input_data_scaled)

#     # Display Result
#     if prediction[0] == 1:
#         st.error("This customer is likely to churn. ❌")
#     else:
#         st.success("This customer is not likely to churn. ✅")


import streamlit as st
import pickle
import numpy as np

# Load the trained model, encoder, and scaler
with open('RFC_Model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('lbl_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Prediction")
st.write("### Enter customer details below to predict churn")

# Sidebar for User Guide
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app predicts whether a customer will churn based on their account details and usage patterns.")
    st.write("Adjust the inputs and click **Predict Churn** to see the result.")

# Organizing input fields in columns
col1, col2 = st.columns(2)

# Input Fields
state = col1.selectbox("🌎 State Code", ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY',
 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA',
 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM',
 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'])
state_encoded = encoder.transform([state])  # Label Encoding

area_code = col2.selectbox("📞 Area Code", ['415', '408', '510'])
account_length = col1.number_input("📆 Account Length (days)", min_value=1, max_value=205, step=1)

voice_plan = col2.radio("📢 Voice Plan", ["No", "Yes"], horizontal=True)
voice_messages = col1.number_input("💬 Voice Messages", min_value=0, max_value=42, step=1)

intl_plan = col2.radio("🌍 International Plan", ["No", "Yes"], horizontal=True)
intl_mins = col1.slider("⏳ International Minutes", min_value=3.3, max_value=17.2, step=0.1)
intl_calls = col2.slider("📞 International Calls", min_value=1, max_value=10, step=1)

day_mins = col1.slider("☀️ Day Minutes", min_value=0.0, max_value=351.5, step=1.0)
day_calls = col2.slider("📞 Day Calls", min_value=0, max_value=165, step=1)

eve_mins = col1.slider("🌙 Evening Minutes", min_value=0.0, max_value=363.7, step=1.0)
eve_calls = col2.slider("📞 Evening Calls", min_value=0, max_value=170, step=1)

night_mins = col1.slider("🌃 Night Minutes", min_value=0.0, max_value=395.0, step=1.0)
night_calls = col2.slider("📞 Night Calls", min_value=0, max_value=175, step=1)

customer_calls = st.slider("📞 Customer Service Calls", min_value=0, max_value=9, step=1)

total_charge = st.slider("💰 Gross Total Charges", min_value=31.0, max_value=87.0, step=0.1)

# Converting categorical values to numerical
area_code = int(area_code)
voice_plan = 1 if voice_plan == "Yes" else 0
intl_plan = 1 if intl_plan == "Yes" else 0

# Prepare input data
input_data = np.array([[float(state_encoded[0]), float(area_code), float(account_length),
                        float(voice_plan), float(voice_messages), float(intl_plan), float(intl_mins), float(intl_calls),
                        float(day_mins), float(day_calls), float(eve_mins), float(eve_calls),
                        float(night_mins), float(night_calls), float(customer_calls), float(total_charge)]])

input_data_scaled = scaler.transform(input_data)

# Predict Button
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_data_scaled)
    
    # Display Result with color-coded message
    if prediction[0] == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")
