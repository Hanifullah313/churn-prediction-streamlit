import streamlit as st
import pandas as pd
import joblib

# Load your trained pipeline model
model = joblib.load("artifacts/model.pkl")

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
    # wants to colums one is for predictiona and probability and other section is for image
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict Churn"):
            add_prediction(input_values)
    with col2:
        st.image("data:image/jpeg;base64,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",
              width=200  )
    # set one image below the prediction button
    st.image("https://www.trentonsystems.com/hs-fs/hubfs/Machine_Learning%20.jpeg?width=8082&name=Machine_Learning%20.jpeg",
         caption="Customer Churn Illustration",  use_container_width=True)

if __name__ == "__main__":
    main()
