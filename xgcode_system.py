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
# Collect user inputs
st.sidebar.header("Input Features")

# 1. First, define the widget to get user input
geography_input = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
geo_map = {"France": 0, "Germany": 1, "Spain": 2}
geography_encoded = geo_map[geography_input]
gender_input = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender_map = {"Female": 0, "Male": 1}
gender_encoded = gender_map[gender_input]
CardType_input=st.sidebar.selectbox("Card Type",["GOLD","PLATINUM","DIAMOND","SILVER"])
CardType_map={"GOLD": 0,"PLATINUM": 1,"DIAMOND": 2,"SILVER": 3}
CardType_encoded=CardType_map[CardType_input]
RowNumber=st.sidebar.number_input("RowNumber",min_value=6,max_value=10000,value=6)
Age = st.sidebar.number_input("Age",min_value=18,max_value=84,value=20)
CreditScore = st.sidebar.number_input("CreditScore",min_value=365,max_value=850,value=430)
Tenure=st.sidebar.number_input("Tenure",min_value=0.00,max_value=10.00,value=0.00) 
Balance=st.sidebar.number_input("Balance",min_value=0.00,max_value=221532.80,value=10000.00)
NumOfProducts=st.sidebar.number_input("NumOfProducts",min_value=1.00,max_value=4.00,value=1.00)
HasCrCard=st.sidebar.number_input("Has a CreditCard",min_value=0,max_value=1,value=1)
IsActiveMember=st.sidebar.number_input("IsActiveMember",min_value=0,max_value=1,value=1)
EstimatedSalary=st.sidebar.number_input("EstimatedSalary",min_value=106.670,max_value=199992.48,value=25000.00)
Complain=st.sidebar.number_input("Complain",min_value=0,max_value=1,value=0)
Satisfaction_Score=st.sidebar.number_input("satisfaction rate",min_value=1.00,max_value=5.00,value=1.00)
Point_Earned=st.sidebar.number_input("Points Earned",min_value=219.00,max_value=1000.00,value=250.00)
# Build a single-row DataFrame matching your model's expected column names

input_data = pd.DataFrame([[Geography,Gender,Card_Type,RowNumber,Age,CreditScore,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Complain,Satisfaction_Score,Point_Earned]],
                          columns=["Geography",
                                   "Gender",
                                   "Card Type",
                                   "RowNumber",
                                   "Age",
                                  "CreditScore",
                                   "Tenure",
                                    "Balance",
                                    "NumOfProducts",
                                    "HasCrCard",
                                     "IsActiveMember",
                                     "EstimatedSalary",
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
