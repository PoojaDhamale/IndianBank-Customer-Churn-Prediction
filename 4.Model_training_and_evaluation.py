import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Model Training & Evaluation",
    layout="wide"
)

# ------------------ HEADER ------------------
st.title("🤖 Model Training & Evaluation")
st.markdown(
    """
    This section explains the **modeling strategy**, **class imbalance handling**, 
    and **performance comparison** that led to the final model selection.
    """
)

st.divider()

# ================== CLASS IMBALANCE ==================
st.header("⚠️ Class Imbalance Problem")

st.markdown(
    """
    The dataset shows a strong **class imbalance**, where churned customers are
    significantly fewer than retained customers.

    - Training directly on imbalanced data biases models toward predicting **non-churn**
    - Accuracy alone becomes misleading
    - Recall for churn customers becomes the key business metric
    """
)

st.divider()

# ================== SMOTE ==================
st.header("🔄 Handling Class Imbalance with SMOTE")

st.markdown(
    """
    **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to:
    
    - Balance churned and non-churned classes
    - Improve the model’s ability to learn churn patterns
    - Avoid bias toward majority class
    """
)

st.divider()

# ================== BASELINE MODEL ==================
st.header("🌲 Baseline Model: Random Forest")

st.markdown(
    """
    **Why Random Forest?**
    - Strong baseline model
    - Handles non-linear relationships
    - Easy to interpret

    **Observed Limitation:**
    - High accuracy on retained customers
    - Poor recall for churned customers
    - Missed a large number of actual churn cases
    """
)

st.markdown(
    """
    📉 **Business Impact:**  
    Missing churn customers means **lost revenue opportunities**.
    """
)

st.divider()

# ================== FINAL MODEL ==================
st.header("🚀 Final Model: XGBoost")

st.markdown(
    """
    **Why XGBoost?**
    - Performs well on structured tabular data
    - Handles class imbalance effectively
    - Captures complex feature interactions

    **Results:**
    - Improved recall for churned customers
    - Better balance between false positives and false negatives
    - Chosen for deployment despite slightly lower accuracy
    """
)

st.divider()

# ================== MODEL PERFORMANCE ==================
st.header("📊 Model Performance Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Final Model", "XGBoost")

with col2:
    st.metric("Accuracy", "63%")

with col3:
    st.metric("Primary Metric", "Recall (Churn)")

st.divider()

# ================== CONFUSION MATRIX ==================
st.subheader("🧮 Confusion Matrix (XGBoost)")

st.markdown(
    "The confusion matrix highlights the model’s ability to correctly identify churned customers."
)

cm = [[7984, 4438],
      [3021, 4557]]

fig, ax = plt.subplots(figsize=(3, 2))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    annot_kws={"size": 8},
    ax=ax
)

ax.set_xlabel("Predicted", fontsize=8)
ax.set_ylabel("Actual", fontsize=8)
ax.set_title("Confusion Matrix – XGBoost", fontsize=9)

ax.tick_params(axis='both', labelsize=7)

# VERY IMPORTANT: prevent Streamlit from stretching it
st.pyplot(fig, use_container_width=False)


# ================== KEY TAKEAWAYS ==================
st.header("📌 Key Takeaways")

st.markdown(
    """
    ✔ Accuracy was **not the primary metric**  
    ✔ Recall for churn customers was prioritized  
    ✔ XGBoost showed better real-world business performance  
    ✔ Model choice aligned with **customer retention goals**
    """
)

# ------------------ FINAL NOTE ------------------
st.divider()

st.info(
    "📌 The selected XGBoost model was used for both batch prediction "
    "and live customer churn prediction in the deployed application."
)
