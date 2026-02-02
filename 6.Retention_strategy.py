import streamlit as st

st.set_page_config(page_title="Retention Strategy", layout="wide")

st.title("📌 Customer Retention Strategy Dashboard")

st.markdown("""
This page translates **churn predictions** into **business retention actions**.
The objective is to reduce customer churn by applying **data-driven strategies**.
""")

st.divider()

# ---------------- RISK SEGMENT INPUT ----------------
st.subheader("🔍 Analyze Customer Risk Level")

churn_prob = st.slider(
    "Select Customer Churn Probability (%)",
    min_value=0,
    max_value=100,
    value=45
) / 100

st.divider()

# ---------------- STRATEGY OUTPUT ----------------
st.subheader("📊 Recommended Retention Strategy")

if churn_prob < 0.30:
    st.success("🟢 **Low Churn Risk Customer**")

    st.markdown("""
    **Recommended Actions:**
    - Cross-sell premium banking products  
    - Offer loyalty reward points  
    - Promote long-term relationship programs  
    - Encourage digital banking usage  

    **Business Goal:**  
    Increase customer lifetime value (CLV)
    """)

elif churn_prob < 0.60:
    st.warning("🟡 **Medium Churn Risk Customer**")

    st.markdown("""
    **Recommended Actions:**
    - Personalized offers (lower fees, cashback)  
    - Proactive customer support calls  
    - Product usage nudges (credit card, savings plans)  
    - Short-term incentives  

    **Business Goal:**  
    Improve engagement and prevent escalation
    """)

else:
    st.error("🔴 **High Churn Risk Customer**")

    st.markdown("""
    **Recommended Actions:**
    - Immediate relationship manager intervention  
    - Fee waivers or interest rate benefits  
    - Customized retention plans  
    - Priority grievance resolution  

    **Business Goal:**  
    Prevent customer exit at all costs
    """)

st.divider()

# ---------------- BUSINESS INSIGHTS ----------------
st.subheader("📈 Key Business Insights")

st.markdown("""
- Customers with **low activity and high balance** are more likely to churn  
- **Low credit score** combined with **multiple products** increases churn risk  
- Active engagement significantly reduces churn probability  

**Conclusion:**  
Retention strategies should be **risk-based, personalized, and timely**.
""")
