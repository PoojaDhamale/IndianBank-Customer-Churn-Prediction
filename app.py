import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="India Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------
st.title("🏦 India Bank Customer Churn Prediction System")

st.markdown("""
This application predicts **customer churn** using **machine learning**
and provides **actionable retention strategies**.

Use the navigation panel on the left to explore each module.
""")

st.divider()

# --------------------------------------------------
# SIDEBAR NAVIGATION (ORDERED)
# --------------------------------------------------
st.sidebar.title("📌 Navigation")

st.sidebar.page_link(
    "pages/1.Overview.py",
    label="1️⃣ Overview",
    icon="📘"
)

st.sidebar.page_link(
    "pages/2.Churn_analysis.py",
    label="2️⃣ Churn Analysis & EDA",
    icon="📊"
)

st.sidebar.page_link(
    "pages/3.feature_Engineering.py",
    label="3️⃣ Feature Engineering",
    icon="🛠️"
)

st.sidebar.page_link(
    "pages/4.Model_training_and_evaluation.py",
    label="4️⃣ Model Training & Evaluation",
    icon="🤖"
)

st.sidebar.page_link(
    "pages/5.Live_prediction.py",
    label="5️⃣ Live Prediction",
    icon="🔮"
)

st.sidebar.page_link(
    "pages/6.Retention_strategy.py",
    label="6️⃣ Retention Strategy",
    icon="🎯"
)

st.sidebar.divider()

# --------------------------------------------------
# FOOTER / SUMMARY
# --------------------------------------------------
st.markdown("""
### ✅ Project Highlights
- Realistic **Indian banking churn dataset**
- **Feature engineering** based on domain logic
- **XGBoost model** with ROC-AUC evaluation
- **Live churn prediction**
- **Business-driven retention strategies**

This project is designed to be **interview-ready and industry aligned**.
""")
