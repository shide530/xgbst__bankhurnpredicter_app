import streamlit as streamlit
import pandas as pd
import numpy as np
import maplotlib.pypplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

model = load_model()

st.set_page_config(page_title="XGBoost Churn Prediction System",
                    layout="wide")

# Collect user inputs
st.sidebar.header("Input Features")
feature_1 = st.sidebar.number_input("Feature 1", value=0.0)
feature_2 = st.sidebar.number_input("Feature 2", value=0.0)

# Build a single-row DataFrame matching your model's expected column names
input_data = pd.DataFrame([[feature_1, feature_2]], columns=["Feature_1", "Feature_2"])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {prediction}")
