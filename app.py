import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Load Model and Encoders
@st.cache_resource
def load_artifacts():
    with open('model\customer_churn.pkl', 'rb') as f:
        model_data = pickle.load(f)
    with open('model\encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model_data['model'], model_data['features_name'], encoders


model, features_name, encoders = load_artifacts()

# App Title & Description
st.title("📊 Telco Customer Churn Prediction Dashboard")
st.write(
    "Is dashboard ke zariye aap customer ki details enter kar ke yeh pata laga sakte hain ke woh churn karega ya nahi.")

st.sidebar.header("Customer Input Parameters")


def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1])
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 1)

    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

    online_security = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])

    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.sidebar.selectbox('Payment Method', [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])

    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', 0.0, 150.0, 29.85)
    total_charges = st.sidebar.number_input('Total Charges ($)', 0.0, 10000.0, 29.85)

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    return pd.DataFrame([data])


input_df = user_input_features()

# Main Panel Display Input Summary
st.subheader("Customer Input Summary")
st.dataframe(input_df)

# Preprocess and Predict Button
if st.button("Predict Churn"):
    # Process input using saved encoders
    processed_df = input_df.copy()
    for col, encoder in encoders.items():
        if col in processed_df.columns:
            # Handle unseen labels gracefully if needed
            val = processed_df[col].iloc[0]
            if val in encoder.classes_:
                processed_df[col] = encoder.transform([val])
            else:
                processed_df[col] = 0  # Default fallback

    # Align features order
    processed_df = processed_df[features_name]

    # Prediction
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ **Prediction: The customer is likely to CHURN.**")
    else:
        st.success(f"✅ **Prediction: The customer is likely to STAY (No Churn).**")

    st.write(f"**Prediction Probability:**")
    st.write(f"- No Churn: `{prediction_proba[0][0] * 100:.2f}%`")
