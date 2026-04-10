import streamlit as st
import pandas as pd
import joblib

model=joblib.load('fraud_detection_pipeline.pkl')
st.title('Fraud Detection prediction System')
st.markdown('please enter your transection details and use the predict button')

st.divider()
transaction_type=st.selectbox('Transaction_type',['PAYMENT','TRANSFER','CASH-OUT','DEPOSIT'])
amount=st.number_input("Amount",min_value=0.0,value=1000.0)
oldbalanceOrg=st.number_input("Old Balance(Sender)",min_value =0.0,value=10000.00)
newbalanceOrig=st.number_input("New Balance (sender)",min_value=0.0,value=90000.00)
oldbalanceDest=st.number_input("Old balance (Recevier)",min_value=0.0,value=0.0)
newbalanceDest=st.number_input("new Balance(Reciver)",min_value=0.0,value=0.0)

# Now we will  nake the prection botton
if st.button("Predict"):
    input_data=pd.DataFrame([{
        'type':transaction_type,
        'amount':amount,
        'oldbalanceOrg':oldbalanceOrg,
        'newbalanceOrig':newbalanceOrig,
        'oldbalanceDest':oldbalanceDest,
        'newbalanceDest':newbalanceDest
    }])
    prediction=model.predict(input_data)[0]

    st.subheader(f"prediction : '{int(prediction)}'")
    if prediction == 1:
        st.error('This Transection Can be Fraud')
    else:
        st.success('This Transection is Not Fraud')