import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page title
st.title("Insurance Claim Prediction")

st.write("Random Forest Model")

# Load dataset
df = pd.read_csv("Insurance_Claim_Prediction_Dataset.csv")

# Encode text columns
encoders = {}

for column in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# Input and output
X = df.drop("Claim_Status", axis=1)
y = df["Claim_Status"]

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

st.write("Enter the details below:")

# Inputs
age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

policy_type = st.selectbox(
    "Policy Type",
    ["Health", "Auto", "Life", "Home"]
)

policy_duration = st.number_input(
    "Policy Duration (Years)",
    1,
    50,
    5
)

annual_premium = st.number_input(
    "Annual Premium",
    0,
    1000000,
    20000
)

annual_income = st.number_input(
    "Annual Income",
    0,
    10000000,
    500000
)

previous_claim_count = st.number_input(
    "Previous Claim Count",
    0,
    50,
    0
)

claim_amount = st.number_input(
    "Claim Amount",
    0,
    5000000,
    100000
)

incident_severity = st.selectbox(
    "Incident Severity",
    ["Low", "Medium", "High"]
)

previous_claim = st.selectbox(
    "Previous Claim",
    ["Yes", "No"]
)

fraud_history = st.selectbox(
    "Fraud History",
    ["Yes", "No"]
)

# Prediction
if st.button("Predict Claim Status"):

    # Convert text inputs to numbers
    gender_value = encoders["Gender"].transform([gender])[0]

    policy_value = encoders["Policy_Type"].transform([policy_type])[0]

    severity_value = encoders["Incident_Severity"].transform(
        [incident_severity]
    )[0]

    previous_claim_value = encoders["Previous_Claim"].transform(
        [previous_claim]
    )[0]

    fraud_value = encoders["Fraud_History"].transform(
        [fraud_history]
    )[0]

    # Create input row
    input_data = pd.DataFrame([[
        0,
        age,
        gender_value,
        policy_value,
        policy_duration,
        annual_premium,
        annual_income,
        previous_claim_count,
        claim_amount,
        severity_value,
        previous_claim_value,
        fraud_value
    ]], columns=X.columns)

    # Prediction
    prediction = model.predict(input_data)

    # Convert number back to text
    result = encoders["Claim_Status"].inverse_transform(
        prediction
    )[0]

    st.subheader("Prediction")

    if result == "Approved":
        st.success("Claim Status: APPROVED")
    else:
        st.error("Claim Status: REJECTED")