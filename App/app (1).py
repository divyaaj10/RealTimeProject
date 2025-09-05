import streamlit as st
import pandas as pd
import joblib

# Load trained churn model
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title(" Customer Churn Prediction App")
st.write("Predict churn for individual customers or upload a CSV for bulk predictions.")

# Sidebar selection
mode = st.sidebar.radio("Choose Mode:", ["Single Prediction", "Bulk Prediction (CSV Upload)"])

# ---------------------- SINGLE PREDICTION ----------------------
if mode == "Single Prediction":
    st.sidebar.header("Customer Information")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

    # Prepare input
    input_data = pd.DataFrame([{
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
        "TotalCharges": total_charges
    }])

    if st.sidebar.button("Predict Churn"):
        churn_prob = model.predict_proba(input_data)[:, 1][0]
        churn_class = model.predict(input_data)[0]  # 0 or 1

        st.subheader(" Prediction Result")
        st.write(f"**Churn Probability:** {churn_prob:.2f}")
        st.write(f"**Churn Prediction:** {'Yes' if churn_class == 1 else 'No'}")

        if churn_class == 1:
            st.error("Customer is likely to churn. Take preventive action!")
        else:
            st.success(" Customer is likely to stay.")

# ---------------------- BULK PREDICTION ----------------------
else:
    st.subheader(" Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data (Before Cleaning)")
        st.dataframe(data.head())

        # -------- Data Cleaning --------
        data = data.replace(" ", pd.NA)  # Convert blank spaces to NaN
        data = data.dropna(how="all")    # Drop rows that are fully empty
        data = data.fillna(0)            # Fill remaining NaN with 0 (or adjust per column meaning)

        # Convert TotalCharges & MonthlyCharges to numeric safely
        for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

        st.write("### Cleaned Data Preview (After Cleaning)")
        st.dataframe(data.head())

        if st.button("Run Bulk Prediction"):
            try:
                predictions = model.predict(data)
                probabilities = model.predict_proba(data)[:, 1]

                data["Churn Prediction"] = ["Yes" if x == 1 else "No" for x in predictions]
                data["Churn Probability"] = probabilities

                st.success(" Predictions Completed!")
                st.write(" Prediction Results")
                st.dataframe(data.head(20))

                # Download results
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=" Download Predictions as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f" Bulk prediction failed: {e}")
