
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('fraud_detection_pipeline.pkl')

# Page config
st.set_page_config(page_title="Fraud Detection", layout="centered")

# 🔥 Custom CSS (makes it look modern)
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}
.stNumberInput input {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💳 Fraud Detection System")
st.markdown("### 🔍 Detect suspicious financial transactions using Machine Learning")

st.divider()

# 🔷 Input Card
st.markdown("## 📥 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "💱 Transaction Type",
        ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT']
    )
    
    amount = st.number_input(
        "💰 Amount",
        min_value=0.0,
        value=1000.0
    )
    
    oldbalanceOrg = st.number_input(
        "👤 Sender Balance (Before)",
        min_value=0.0,
        value=10000.0
    )

with col2:
    newbalanceOrig = st.number_input(
        "👤 Sender Balance (After)",
        min_value=0.0,
        value=9000.0
    )
    
    oldbalanceDest = st.number_input(
        "🏦 Receiver Balance (Before)",
        min_value=0.0,
        value=0.0
    )
    
    newbalanceDest = st.number_input(
        "🏦 Receiver Balance (After)",
        min_value=0.0,
        value=0.0
    )

st.divider()

# 🔷 Predict Section
if st.button("🚀 Analyze Transaction", use_container_width=True):

    # Input dataframe
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.markdown("## 📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        if prediction == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ SAFE TRANSACTION")

    with colB:
        st.metric("Fraud Probability", f"{proba:.2%}")

    st.progress(int(proba * 100))

    # 🔥 Smart insight
    st.markdown("### 🧠 Model Insight")
    if proba > 0.7:
        st.warning("High risk transaction pattern detected.")
    elif proba > 0.4:
        st.info("Moderate risk. Needs verification.")
    else:
        st.success("Low risk transaction.")
