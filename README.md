🏦 India Bank Customer Churn Prediction System
📌 Project Overview

This is an end-to-end Machine Learning project that predicts whether a bank customer is likely to churn (leave the bank) and provides actionable retention strategies.

Focused on a realistic Indian banking use case.

Built for placement-ready portfolio, showcasing both technical ML skills and business understanding.

Deployed as an interactive Streamlit web application.

🎯 Problem Statement

Customer churn is a major challenge for banks. Retaining existing customers is cheaper than acquiring new ones.

Objectives of this project:

Identify customers likely to churn

Understand key factors driving churn

Predict churn probability in real time

Suggest targeted retention strategies

🧠 Machine Learning Pipeline
1️⃣ Data Collection

Dataset: Indian bank customers

Features: Customer demographics, financial attributes, behavioral data

2️⃣ Exploratory Data Analysis (EDA)

Churn distribution analysis

Feature-wise churn trends

Visualizations: bar plots, box plots

3️⃣ Feature Engineering

Balance-to-Salary ratio

Tenure × Number of Products

Low Credit Score flag

High Balance & Low Activity flag

4️⃣ Data Preprocessing

Numerical feature scaling: StandardScaler

Categorical feature encoding: OneHotEncoder

Consistent feature alignment using a saved feature list

5️⃣ Model Training

Algorithm: XGBoost Classifier

Train-test split with stratification

6️⃣ Model Evaluation

Accuracy

Confusion Matrix

Classification Report

ROC-AUC Score

7️⃣ Deployment

Interactive Streamlit web app

Live churn prediction

Business-driven retention strategies

🛠️ Tech Stack

Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost

Deployment: Streamlit

Model Serialization: Pickle

📂 Project Structure
IndiaBankChurnPrediction/
│
├── app.py
├── xgb_model.pkl
├── scaler.pkl
├── encoder.pkl
├── model_features.pkl
├── indian_bank_customer_churn.csv
├── X_test.csv
├── y_test.csv
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Churn_Analysis.py
│   ├── 3_Feature_Engineering.py
│   ├── 4_Model_Training_Evaluation.py
│   ├── 5_Live_Prediction.py
│   └── 6_Retention_Strategy.py
├── EDA.ipynb
└── final_indian_customer_churn_prediction.ipynb

🚀 How to Run
Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Run Streamlit App
streamlit run app.py

🔮 Live Prediction Module

Input customer details (age, balance, credit score, geography, etc.)

Automatic preprocessing (scaling & encoding)

Real-time churn probability prediction

Display results clearly:

Customer likely to churn

Customer likely to stay

🎯 Retention Strategy Module

Based on churn risk, the app suggests:

Personalized offers for high-risk customers

Engagement strategies for inactive users

Credit-based financial counseling

Product simplification for overloaded customers

Loyalty & reward programs for medium-risk users

Bridges ML predictions with real business decisions.

📊 Key Insights

High balance but low activity → more likely to churn

Low credit score → higher churn probability

Customers with 4 products → churn more due to complexity

Active members → lower churn risk

📈 Future Enhancements

Model explainability using SHAP

Batch prediction via CSV upload

Database integration (MySQL / PostgreSQL)

Cloud deployment (AWS / Streamlit Cloud)

👩‍💻 Author

Pooja Dhamale
Machine Learning & Data Science Enthusiast
