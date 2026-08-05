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
Age = st.sidebar.number_input("Age",min_value=18,max_value=84,value=20)
CreditScore = st.sidebar.number_input("CreditScore",min_value=365,max_value=850,value=430)
Tenure=st.sidebar.number_input("Tenure",min_value=0.00,max_value=10.00,value=0.00) 
Balance=st.sidebar.number_input("Balance",min_value=0.00,max_value=221532.80,value=10000.00)
NumOfProducts=st.sidebar.number_input("NumOfProducts",min_value=1.00,max_value=4.00,value=1.00)
HasCrCard=st.sidebar.number_input("Has a CreditCard",min_value=0,max_value=1,value=1)
EstimatedSalary=st.sidebar.number_input("EstimatedSalary",min_value=106.670,max_value=199992.48,value=25000.00)
Complain=st.sidebar.number_input("Complain",min_value=0,max_value=1,value=0)
Satisfaction_Score=st.sidebar.number_input("satisfaction rate",min_value=1.00,max_value=5.00,value=1.00)
Point_Earned=st.sidebar.number_input("Points Earned",min_value=219.00,max_value=1000.00,value=250.00)
# Build a single-row DataFrame matching your model's expected column names

input_data = pd.DataFrame([[Age,CreditScore,Tenure,Balance,NumOfProducts,HasCrCard,EstimatedSalary,complain,Satisfaction_Score,Points_Earned]],
                          columns=["Age"
                                  "CreditScore",
                                   "Tenure",
                                    "Balance",
                                    "NumOfProducts",
                                    "HasCrCard",
                                     "IsActiveMember",
                                     "EstimatedSalary,"
                                      "Complain",
                                      "Satisfaction Score",
                                      "Point Earned"]
                         )

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction==1:
      result="Customer is likely to churn" 
    else:
      result="Customer is likely to stay"
    st.success(f"Prediction: {result}")
