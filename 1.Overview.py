import streamlit as st
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Overview | Bank Customer Churn Prediction",
    layout="wide"
)

# ------------------ LOAD RAW DATA ------------------
@st.cache_data
def load_data():
    # 🔁 Update path if required
    return pd.read_csv(r"C:\Users\dhama\Desktop\IndiaBankChurnPrediction\indian_bank_customer_churn.csv")

df = load_data()

# ------------------ KPI CALCULATIONS ------------------
total_customers = df.shape[0]
churned_customers = df[df["Churn"] == 1].shape[0]
retained_customers = df[df["Churn"] == 0].shape[0]

active_customers = df[df["Is_Active_Member"] == 1].shape[0]
inactive_customers = df[df["Is_Active_Member"] == 0].shape[0]

churn_rate = round((churned_customers / total_customers) * 100, 2)

# ------------------ HEADER ------------------
st.title("🏦 AI-Powered Bank Customer Churn Prediction")
st.subheader("End-to-End Machine Learning Application")

st.markdown(
    """
    This application predicts whether a bank customer is likely to **churn (leave the bank)** 
    based on their demographic, financial, and behavioral information.
    """
)

st.divider()

# ------------------ PROBLEM STATEMENT ------------------
st.header("📌 Problem Statement")

st.markdown(
    """
    Customer churn is a major challenge for banks, as acquiring new customers is significantly 
    more expensive than retaining existing ones.

    **Objective:**
    - Identify customers who are at high risk of churn  
    - Enable proactive customer retention strategies  
    - Improve customer lifetime value using data-driven insights  
    """
)

# ------------------ DATASET KPIs ------------------
st.header("📊 Dataset KPIs (Raw Data)")

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("👥 Total Customers", f"{total_customers:,}")

with k2:
    st.metric("❌ Churned Customers", f"{churned_customers:,}")

with k3:
    st.metric("✅ Retained Customers", f"{retained_customers:,}")

with k4:
    st.metric("🟢 Active Customers", f"{active_customers:,}")

with k5:
    st.metric("📉 Churn Rate (%)", f"{churn_rate}%")

sub1, sub2 = st.columns(2)

with sub1:
    st.metric("⚪ Inactive Customers", f"{inactive_customers:,}")

with sub2:
    st.metric("🧾 Raw Features", "13")

st.markdown(
    """
    ✅ This overview uses **only raw, original customer data** collected by the bank.  
    Feature engineering and transformations are applied **later in the ML pipeline**.
    """
)

st.markdown(
    """
    **Feature Categories:**
    - 🧍 Demographic: Age, Tenure  
    - 💰 Financial: Balance, Estimated Salary  
    - 🏦 Behavioral: Active Member, Credit Card Usage  
    """
)

# ------------------ ML PIPELINE ------------------
st.header("⚙️ Machine Learning Pipeline")

st.markdown(
    """
    ```
    Data Cleaning
        ↓
    Feature Engineering
        ↓
    Model Training
        ↓
    Model Evaluation
        ↓
    Deployment & Prediction
    ```
    """
)

# ------------------ MODELS USED ------------------
st.header("🤖 Models Used")

m1, m2 = st.columns(2)

with m1:
    st.markdown(
        """
        **🌲 Random Forest (Baseline Model)**  
        - Used as an initial benchmark  
        - Strong performance on non-churn customers  
        - Lower recall for churn class  
        """
    )

with m2:
    st.markdown(
        """
        **⚡ XGBoost (Final Model)**  
        - Handles class imbalance effectively  
        - Improved churn customer detection  
        - Selected for final deployment  
        """
    )

# ------------------ KEY RESULTS ------------------
st.header("📈 Key Results")

r1, r2, r3 = st.columns(3)

with r1:
    st.metric("✅ Accuracy", "63%")

with r2:
    st.metric("🔍 Churn Detection", "Improved")

with r3:
    st.metric("🎯 Business Focus", "Recall over Accuracy")

# ------------------ APPLICATION FEATURES ------------------
st.header("🚀 Application Capabilities")

st.markdown(
    """
    ✔ 📊 Interactive churn analysis & visual insights  
    ✔ 🧠 Feature importance visualization  
    ✔ 📂 CSV upload for batch churn prediction  
    ✔ ⚡ Live churn prediction for new customers  
    ✔ 🎨 Clean & intuitive Streamlit dashboard  
    """
)

# ------------------ NAVIGATION NOTE ------------------
st.divider()

st.info(
    "📍 Use the sidebar to navigate through churn analysis, feature engineering, "
    "model evaluation, and live prediction modules."
)
