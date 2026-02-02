import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD ARTIFACTS ----------------
with open(r"C:\Users\dhama\Desktop\IndiaBankChurnPrediction\xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\dhama\Desktop\IndiaBankChurnPrediction\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(r"C:\Users\dhama\Desktop\IndiaBankChurnPrediction\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open(r"C:\Users\dhama\Desktop\IndiaBankChurnPrediction\model_features.pkl", "rb") as f:
    final_feature_names = pickle.load(f)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🔮 Live Churn Prediction",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "Enter customer details below to **predict churn risk in real time**."
)

st.divider()

# ---------------- USER INPUT ----------------
st.subheader("📋 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("🎂 Age", 18, 100, 35)
    tenure = st.number_input("📆 Tenure (Years)", 0, 20, 5)
    credit_score = st.number_input("💳 Credit Score", 300, 900, 650)

with col2:
    balance = st.number_input("💰 Account Balance (INR)", 0.0, 10_000_000.0, 50000.0)
    salary = st.number_input("💼 Estimated Salary (INR)", 0.0, 10_000_000.0, 600000.0)
    num_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4])

with col3:
    has_cc = st.selectbox("💳 Has Credit Card?", ["Yes", "No"])
    active = st.selectbox("⚡ Is Active Member?", ["Yes", "No"])
    gender = st.selectbox("🧑 Gender", ["Male", "Female"])

state = st.selectbox(
    "📍 State",
    ["Gujarat", "Karnataka", "Maharashtra", "Tamil Nadu",
     "Telangana", "Uttar Pradesh", "West Bengal"]
)

account_type = st.selectbox("🏦 Account Type", ["Savings", "Salary"])

# ---------------- FEATURE ENGINEERING ----------------
has_cc = 1 if has_cc == "Yes" else 0
active = 1 if active == "Yes" else 0

balance_to_salary = balance / salary if salary > 0 else 0
tenure_numproducts = tenure * num_products
low_credit_score = 1 if credit_score < 600 else 0
highbalance_lowactivity = 1 if (balance > 100000 and active == 0) else 0

# ---------------- CREATE INPUT DF ----------------
input_df = pd.DataFrame({
    "Age": [age],
    "Tenure_Years": [tenure],
    "Balance_INR": [balance],
    "Num_Products": [num_products],
    "Has_Credit_Card": [has_cc],
    "Is_Active_Member": [active],
    "Estimated_Salary_INR": [salary],
    "Credit_Score": [credit_score],
    "Balance_to_Salary": [balance_to_salary],
    "Tenure_NumProducts": [tenure_numproducts],
    "Low_Credit_Score": [low_credit_score],
    "HighBalance_LowActivity": [highbalance_lowactivity],
    "Gender": [gender],
    "State": [state],
    "Account_Type": [account_type]
})

# ---------------- PREPROCESS ----------------
num_cols = [
    "Age", "Tenure_Years", "Balance_INR", "Num_Products",
    "Has_Credit_Card", "Is_Active_Member",
    "Estimated_Salary_INR", "Credit_Score",
    "Balance_to_Salary", "Tenure_NumProducts",
    "Low_Credit_Score", "HighBalance_LowActivity"
]

cat_cols = ["Gender", "State", "Account_Type"]

X_num = scaler.transform(input_df[num_cols])
X_cat = encoder.transform(input_df[cat_cols])

X_final = np.concatenate([X_num, X_cat], axis=1)
X_final = pd.DataFrame(X_final, columns=final_feature_names)

# ---------------- PREDICTION ----------------
st.divider()

if st.button("🔍 Predict Churn", use_container_width=True):
    with st.spinner("Analyzing customer behavior..."):
        prob = model.predict_proba(X_final)[0][1]
        prediction = model.predict(X_final)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk**\n\nProbability: **{prob:.2%}**")
    else:
        st.success(f"✅ **Low Churn Risk**\n\nRetention Probability: **{1 - prob:.2%}**")

    st.progress(int(prob * 100))

    if prob > 0.7:
        st.warning("📢 Recommended Action: Immediate retention strategy required.")
    elif prob > 0.4:
        st.info("📌 Recommended Action: Monitor customer engagement closely.")
    else:
        st.success("🎉 Customer is stable. No immediate action needed.")
