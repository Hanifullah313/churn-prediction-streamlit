import streamlit as st
import pandas as pd
import joblib

# Load your trained pipeline model
model = joblib.load(r"C:\Users\Hanif ullah laptop\Desktop\Churn Prediction\artifacts\model.pkl")

def add_sidebar():
    st.sidebar.header("Customer Information")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [1, 0])
    partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.slider("Total Charges", 0.0, 8000.0, 1000.0)

    # Collect all inputs
    inputs_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
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
        "TotalCharges": total_charges
    }

    return inputs_dict


def add_prediction(input_values):
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([input_values])

    # Directly predict using the pipeline model
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Display prediction probability through a progress bar
    st.write("### Prediction Probability")
    st.progress(prob)
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to **CHURN** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The customer is likely to **STAY** (Probability: {prob:.2f})")

def main():
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
    st.title("📊 Customer Churn Prediction App")

    st.write("""
    This app predicts whether a customer is likely to **churn** (leave the service) 
    or **stay**, based on their account and service information.
    """)

    input_values = add_sidebar()

    if st.button("Predict Churn"):
        add_prediction(input_values)


if __name__ == "__main__":
    main()
