import streamlit as st
import pandas as pd
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthSaaS Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

# --- App Title ---
st.title("SynthSaaS Customer Churn Prediction Dashboard")
st.markdown("This dashboard allows you to predict churn for a customer in real-time and view key model insights.")

# --- Sidebar for User Input ---
st.sidebar.header("Predict Churn for a New Customer")

# Define input fields
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
senior_citizen = st.sidebar.selectbox("Senior Citizen", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
partner = st.sidebar.selectbox("Has Partner", ("Yes", "No"))
dependents = st.sidebar.selectbox("Has Dependents", ("Yes", "No"))
phone_service = st.sidebar.selectbox("Has Phone Service", ("Yes", "No"))
multiple_lines = st.sidebar.selectbox("Has Multiple Lines", ("Yes", "No", "No phone service"))
internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
online_security = st.sidebar.selectbox("Has Online Security", ("Yes", "No", "No internet service"))
online_backup = st.sidebar.selectbox("Has Online Backup", ("Yes", "No", "No internet service"))
device_protection = st.sidebar.selectbox("Has Device Protection", ("Yes", "No", "No internet service"))
tech_support = st.sidebar.selectbox("Has Tech Support", ("Yes", "No", "No internet service"))
streaming_tv = st.sidebar.selectbox("Has Streaming TV", ("Yes", "No", "No internet service"))
streaming_movies = st.sidebar.selectbox("Has Streaming Movies", ("Yes", "No", "No internet service"))
contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
paperless_billing = st.sidebar.selectbox("Uses Paperless Billing", ("Yes", "No"))
payment_method = st.sidebar.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=500.0)


# --- Prediction Logic ---
if st.sidebar.button("Predict Churn"):
    # Create a dictionary from the inputs
    customer_data = {
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    try:
        # Send data to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json=customer_data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        prediction_result = response.json()
        
        # Display the prediction
        st.subheader("Prediction Result")
        churn_prob = prediction_result['churn_probability']
        
        if prediction_result['prediction'] == 'Churn':
            st.error(f"Prediction: **Churn** (Probability: {churn_prob:.2%})")
            st.warning("This customer is at high risk of churning. Consider taking retention actions.")
        else:
            st.success(f"Prediction: **No Churn** (Probability of Churn: {churn_prob:.2%})")
            st.info("This customer is at low risk of churning.")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the prediction API. Please ensure the FastAPI server is running. Error: {e}")

# --- Key Business Insights Section ---
st.markdown("---")
st.header("Key Business Insights")
st.markdown("""
Our analysis has revealed several key factors that strongly influence customer churn. 
These insights are critical for developing targeted retention strategies.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Churn Drivers")
    st.markdown("""
    Based on our model (Logistic Regression, ROC-AUC: 0.82), the following are the most powerful predictors of churn:
    - **Contract Type:** `Month-to-month` contracts are the single biggest risk factor.
    - **Tenure:** Shorter tenure strongly correlates with higher churn risk.
    - **Internet Service:** Customers with `Fiber optic` service are more likely to churn.
    - **Payment Method:** `Electronic check` is associated with a higher churn rate.
    """)

with col2:
    st.subheader("Actionable Recommendations")
    st.markdown("""
    - **Targeted Campaigns:** Proactively offer yearly contracts to high-value customers on month-to-month plans, especially if they are new.
    - **Onboarding:** Enhance the onboarding experience for new customers in their first 1-3 months to improve retention.
    - **Service Investigation:** Investigate potential issues with the Fiber Optic service. Are there pricing, reliability, or support concerns?
    - **Payment Incentives:** Encourage customers to switch from electronic checks to automatic payment methods.
    """)

# To run this dashboard:
# 1. Ensure the FastAPI server is running in another terminal (`uvicorn api.main:app`).
# 2. Navigate to the project root in a new terminal.
# 3. Run the command: streamlit run dashboard/main.py
