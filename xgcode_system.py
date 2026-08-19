import streamlit as streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="XGBoost Churn Prediction System",
                    layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")
model = load_model()
# Collect user inputs
st.sidebar.header("Input Features")

RowNumber=st.sidebar.number_input("RowNumber",min_value=6,max_value=10000,value=6)
CreditScore = st.sidebar.number_input("CreditScore",min_value=365,max_value=850,value=430)
Geography_input = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
Geo_map = {"France": 0, "Germany": 1, "Spain": 2}
Geography_encoded = Geo_map[Geography_input]
Gender_input = st.sidebar.selectbox("Gender", ["Female", "Male"])
Gender_map = {"Female": 0, "Male": 1}
Gender_encoded = Gender_map[Gender_input]
Age = st.sidebar.number_input("Age",min_value=18,max_value=84,value=20)
Tenure=st.sidebar.number_input("Tenure",min_value=0.00,max_value=10.00,value=0.00) 
Balance=st.sidebar.number_input("Balance",min_value=0.00,max_value=221532.80,value=10000.00)
NumOfProducts=st.sidebar.number_input("NumOfProducts",min_value=1.00,max_value=4.00,value=1.00)
HasCrCard=st.sidebar.number_input("Has a CreditCard",min_value=0,max_value=1,value=1)
IsActiveMember=st.sidebar.number_input("IsActiveMember",min_value=0,max_value=1,value=1)
EstimatedSalary=st.sidebar.number_input("EstimatedSalary",min_value=106.670,max_value=199992.48,value=25000.00)
Complain=st.sidebar.number_input("Complain",min_value=0,max_value=1,value=0)
Satisfaction_Score=st.sidebar.number_input("satisfaction rate",min_value=1.00,max_value=5.00,value=1.00)
CardType_input=st.sidebar.selectbox("Card Type",["GOLD","PLATINUM","DIAMOND","SILVER"])
CardType_map={"GOLD": 0,"PLATINUM": 1,"DIAMOND": 2,"SILVER": 3}
CardType_encoded=CardType_map[CardType_input]
Point_Earned=st.sidebar.number_input("Points Earned",min_value=219.00,max_value=1000.00,value=250.00)
# Build a single-row DataFrame matching your model's expected column names
feature_columns=["RowNumber",
                 "CreditScore",
                 "Geography",
                "Gender",
                 "Age"
               "Tenure",
               "Balance",
               "NumOfProducts",
               "HasCrCard",
               "IsActiveMember",
               "EstimatedSalary",
               "Complain",
               "Satisfaction Score",
               "Card Type",
              "Point Earned"]
                         
input_data = pd.DataFrame([[RowNumber,CreditScore,Geography_encoded,Gender_encoded,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Complain,Satisfaction_Score,CardType_encoded,Point_Earned]],columns=feature_columns)

if st.button("Predict"):
    st.write("Running predictions...")
    prediction = model.predict_proba(input_data)[0]
    churn_prob = prediction[1] * 100
    
    # 2. Main Dashboard - Metrics Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Churn Probability", value=f"{churn_prob:.1f}%")
    with col2:
        st.metric(label="Retention Probability", value=f"{(100 - churn_prob):.1f}%")
    with col3:
        status = "High Risk" if churn_prob >= 50 else "Low Risk"
        st.metric(label="Risk Tier", value=status)

    st.divider()

    # 3. Main Dashboard - Visualizations Section
    left_chart_col, right_chart_col = st.columns(2)

    with left_chart_col:
        st.subheader("Risk Score Gauge")
        st.write("Current Churn Probability Indicator:")
        st.progress(int(churn_prob))
        
        if churn_prob >= 50:
            st.error("⚠️ Action Needed: High risk of customer cancellation.")
        else:
            st.success("✅ Customer profile appears stable.")

    with right_chart_col:
        st.subheader("Model Feature Importances")
        importances = model.feature_importances_

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(feature_columns, importances, color="#4F8BF9")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        
        st.pyplot(fig)
else:
    st.info("Adjust the sidebar parameters and click **Run Prediction** to view the report.")
