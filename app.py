"""
app.py
------
Streamlit web app for the Customer Churn Prediction System.
Loads the trained pipeline (models/churn_model.pkl) and lets a user
enter a customer's profile to get a real-time churn prediction.

Run with:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


model = load_model()

st.title("📉 Customer Churn Prediction")
st.write(
    "Enter a customer's profile below to predict whether they are likely "
    "to churn. Powered by a tuned ML pipeline trained on telecom customer data."
)

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, step=1.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **churn**. (Confidence: {probability:.1%})")
    else:
        st.success(f"✅ This customer is likely to **stay**. (Churn probability: {probability:.1%})")

    st.progress(min(max(probability, 0.0), 1.0))

st.divider()
st.caption(
    "Model: tuned scikit-learn pipeline (see train_model.py) · "
    "Dataset: Telco-style customer data · Project by Nancharu Harsha"
)
