import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Churn Analysis & Visuals",
    layout="wide"
)

# ------------------ LOAD RAW DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\Users\dhama\Desktop\IndiaBankChurnPrediction\indian_bank_customer_churn.csv"
    )

df = load_data()

# Map churn labels for readability
df["Churn_Label"] = df["Churn"].map({0: "Retained", 1: "Churned"})

# ------------------ HEADER ------------------
st.title("📊 Churn Analysis & Visual Insights")
st.markdown(
    """
    This section explores **customer churn patterns** using the **raw dataset only**.
    The objective is to understand customer behavior *before* applying feature engineering
    or machine learning models.
    """
)

st.divider()

# ================== 1️⃣ OVERALL CHURN DISTRIBUTION ==================
st.header("1️⃣ Overall Churn Distribution")

churn_counts = df["Churn_Label"].value_counts().reset_index()
churn_counts.columns = ["Churn Status", "Customer Count"]

fig_churn = px.pie(
    churn_counts,
    names="Churn Status",
    values="Customer Count",
    hole=0.4,
    title="Churn vs Retained Customers"
)

st.plotly_chart(fig_churn, use_container_width=True)

st.markdown(
    "🔍 **Insight:** The dataset is imbalanced, with churned customers forming a smaller proportion of the total population."
)

st.divider()

# ================== 2️⃣ CHURN BY GENDER ==================
st.header("2️⃣ Churn by Gender")

fig_gender = px.histogram(
    df,
    x="Gender",
    color="Churn_Label",
    barmode="group",
    title="Churn Distribution by Gender",
    labels={"Churn_Label": "Customer Status"}
)

fig_gender.update_layout(
    xaxis_title="Gender",
    yaxis_title="Customer Count",
    legend_title_text="Status"
)

st.plotly_chart(fig_gender, use_container_width=True)

st.markdown(
    "🔍 **Insight:** Churn behavior varies slightly across genders, indicating demographic influence."
)

st.divider()

# ================== 3️⃣ CHURN BY GEOGRAPHY ==================
st.header("3️⃣ Churn by Geography")

fig_geo = px.histogram(
    df,
    x="State",
    color="Churn_Label",
    barmode="group",
    title="Churn Distribution by Geography",
    labels={"Churn_Label": "Customer Status"}
)

fig_geo.update_layout(
    xaxis_title="State",
    yaxis_title="Customer Count",
    legend_title_text="Status"
)

st.plotly_chart(fig_geo, use_container_width=True)

st.markdown(
    "🔍 **Insight:** Certain states show a higher concentration of churned customers, suggesting regional patterns."
)

st.divider()

# ================== 4️⃣ CHURN VS ACTIVE MEMBER ==================
st.header("4️⃣ Churn vs Active Membership")

fig_active = px.histogram(
    df,
    x="Is_Active_Member",
    color="Churn_Label",
    barmode="group",
    title="Churn vs Active Membership",
    text_auto=True,
    labels={"Churn_Label": "Customer Status"}
)

fig_active.update_layout(
    xaxis_title="Active Member (0 = No, 1 = Yes)",
    yaxis_title="Customer Count",
    legend_title_text="Status"
)

st.plotly_chart(fig_active, use_container_width=True)

st.markdown(
    "🔍 **Insight:** Inactive customers show a significantly higher tendency to churn."
)

st.divider()

# ================== 5️⃣ CREDIT SCORE VS CHURN ==================
st.header("5️⃣ Credit Score vs Churn")

fig_credit = px.box(
    df,
    x="Churn_Label",
    y="Credit_Score",
    title="Credit Score Distribution by Churn Status",
    labels={"Churn_Label": "Customer Status"}
)

st.plotly_chart(fig_credit, use_container_width=True)

st.markdown(
    "🔍 **Insight:** Churned customers tend to have a lower median credit score compared to retained customers."
)

st.divider()

# ================== 6️⃣ ESTIMATED SALARY VS CHURN ==================
st.header("6️⃣ Estimated Salary vs Churn")

fig_salary = px.box(
    df,
    x="Churn_Label",
    y="Estimated_Salary_INR",
    title="Estimated Salary Distribution by Churn Status",
    labels={"Churn_Label": "Customer Status"}
)

st.plotly_chart(fig_salary, use_container_width=True)

st.markdown(
    "🔍 **Insight:** Estimated salary alone does not strongly differentiate churn behavior."
)

# ------------------ FINAL NOTE ------------------
st.divider()

st.info(
    "📌 These exploratory insights directly informed feature engineering choices "
    "and model selection in later stages of the churn prediction pipeline."
)
