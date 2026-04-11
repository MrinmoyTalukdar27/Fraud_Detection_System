
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('fraud_detection_pipeline.pkl')

# Page config
st.set_page_config(page_title="Fraud Detection", layout="centered")

# Title
st.title("💳 Fraud Detection System")
st.markdown("Enter transaction details below to check if it's **fraudulent or not**.")

st.divider()

# Input Section
st.subheader("📥 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT']
    )
    
    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=1000.0
    )
    
    oldbalanceOrg = st.number_input(
        "Sender Balance Before",
        min_value=0.0,
        value=10000.0
    )

with col2:
    newbalanceOrig = st.number_input(
        "Sender Balance After",
        min_value=0.0,
        value=9000.0
    )
    
    oldbalanceDest = st.number_input(
        "Receiver Balance Before",
        min_value=0.0,
        value=0.0
    )
    
    newbalanceDest = st.number_input(
        "Receiver Balance After",
        min_value=0.0,
        value=0.0
    )

st.divider()

# Predict Button
if st.button("🔍 Predict", use_container_width=True):
    
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])
    
    prediction = model.predict(input_data)[0]
    
    st.subheader("📊 Prediction Result")
    
    if prediction == 1:
        st.error("🚨 This transaction is likely **FRAUD**")
    else:
        st.success("✅ This transaction is **NOT Fraud**")