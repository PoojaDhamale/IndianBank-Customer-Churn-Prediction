import streamlit as st
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Feature Engineering & Preprocessing",
    layout="wide"
)

# ------------------ LOAD FEATURED DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\Users\dhama\Desktop\IndiaBankChurnPrediction\indian_bank_customer_churn_featured1.csv"
    )

df = load_data()

# ------------------ HEADER ------------------
st.title("🧠 Feature Engineering & Preprocessing")
st.markdown(
    """
    This section explains the **feature engineering techniques** applied to the raw dataset
    to improve model performance and capture hidden customer behavior patterns.
    """
)

st.divider()

# ================== WHY FEATURE ENGINEERING ==================
st.header("📌 Why Feature Engineering?")

st.markdown(
    """
    Raw customer attributes often fail to fully represent **customer risk and behavior**.
    Feature engineering helps:
    
    - Capture **financial stress**
    - Represent **customer engagement**
    - Flag **high-risk churn patterns**
    - Improve model learning capability
    """
)

st.divider()

# ================== ENGINEERED FEATURES ==================
st.header("⚙️ Engineered Features")

st.markdown(
    """
    The following new features were created from raw attributes:
    """
)

st.markdown(
    """
    **1️⃣ Balance_to_Salary**  
    - Ratio of account balance to estimated salary  
    - Indicates financial pressure or underutilization  

    **2️⃣ Tenure_NumProducts**  
    - Combines tenure and number of bank products  
    - Represents customer engagement strength  

    **3️⃣ Low_Credit_Score**  
    - Binary flag where Credit Score < 600  
    - Identifies high credit risk customers  

    **4️⃣ HighBalance_LowActivity**  
    - Flags customers with high balance but inactive behavior  
    - Strong churn risk indicator
    """
)

st.divider()

# ================== FEATURE COUNT ==================
st.header("📊 Feature Count Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Original Features", 13)

with col2:
    st.metric("Engineered Features Added", 4)

with col3:
    st.metric("Total Features After Engineering", df.shape[1])

st.divider()

# ================== SAMPLE DATA ==================
st.header("🔍 Sample of Engineered Dataset")

st.markdown(
    "Below is a preview of the dataset after feature engineering:"
)

st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ================== TARGET & FEATURES ==================
st.header("🎯 Feature & Target Separation")

st.markdown(
    """
    - **Target Variable:** `Churn`  
    - **Feature Matrix (X):** All customer attributes excluding churn  
    - This dataset was then used for:
        - Model training & evaluation
    """
)

# ------------------ FINAL NOTE ------------------
st.divider()

st.info(
    "📌 Feature engineering played a key role in improving churn detection, "
    "especially for identifying high-risk customers in the XGBoost model."
)
