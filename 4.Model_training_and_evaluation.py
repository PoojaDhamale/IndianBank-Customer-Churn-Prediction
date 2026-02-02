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
    This section explains the **modeling strategy**, **class imbalance considerations**, 
    and **hyperparameter tuning** that led to the final model selection.
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

# ================== HYPERPARAMETER TUNING ==================
st.header("🔧 Hyperparameter Tuning for XGBoost")

st.markdown(
    """
    **Why hyperparameter tuning?**  
    - Optimizes model performance for both accuracy and recall  
    - Helps prevent overfitting and underfitting  
    - Improves detection of churned customers

    **Parameters tuned include:**  
    - `n_estimators` (number of trees)  
    - `learning_rate` (step size for boosting)  
    - `max_depth` (tree depth)  
    - `subsample` & `colsample_bytree` (row/feature sampling)  
    - `reg_alpha` & `reg_lambda` (regularization)  

    RandomizedSearchCV was used to efficiently search the parameter space and select the best combination.
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

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Final Model", "XGBoost")

with col2:
    st.metric("Accuracy", "49.5%")  # updated

with col3:
    st.metric("Recall (Churn)", "90%")  # class 1 recall

with col4:
    st.metric("Precision (Churn)", "42%")  # class 1 precision

st.divider()

# ================== CONFUSION MATRIX ==================
st.subheader("🧮 Confusion Matrix (XGBoost)")

st.markdown(
    """
    The confusion matrix highlights the model’s ability to correctly identify churned customers.
    
    **Interpretation:**
    - True Positives (TP): 6845 → correctly predicted churn
    - False Negatives (FN): 733 → churn missed
    - True Negatives (TN): 3049 → correctly predicted non-churn
    - False Positives (FP): 9373 → non-churn predicted as churn
    """
)

# updated confusion matrix
cm = [[3049, 9373],
      [733, 6845]]

fig, ax = plt.subplots(figsize=(4, 3))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    annot_kws={"size": 10},
    ax=ax
)

ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)
ax.set_title("Confusion Matrix – XGBoost", fontsize=12)
ax.tick_params(axis='both', labelsize=9)

st.pyplot(fig, use_container_width=False)

# ================== KEY TAKEAWAYS ==================
st.header("📌 Key Takeaways")

st.markdown(
    """
    ✔ Accuracy is 49.5% – lower because the focus is on detecting churn  
    ✔ Recall for churn customers prioritized → 90% of actual churners caught  
    ✔ Precision for churn is 42% → some false positives expected  
    ✔ F1-score balances precision and recall (class 1: 0.58)  
    ✔ Hyperparameter tuning significantly improved performance  
    ✔ XGBoost was chosen for **real-world customer retention goals**
    """
)

# ------------------ FINAL NOTE ------------------
st.divider()

st.info(
    "📌 The selected XGBoost model with optimized hyperparameters is used "
    "for both batch prediction and live customer churn prediction in the deployed application."
)
