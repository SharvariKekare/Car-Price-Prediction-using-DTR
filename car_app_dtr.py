import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("decision_tree_model.pkl")

# Load Dataset
df = pd.read_csv("final_cleaned_dataset.csv")

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction")
st.write("### Decision Tree Regression")

# Dropdown Menus
car_name = st.selectbox(
    "Select Car Name",
    sorted(df["name"].unique())
)

company = st.selectbox(
    "Select Company",
    sorted(df["company"].unique())
)

fuel_type = st.selectbox(
    "Select Fuel Type",
    sorted(df["fuel_type"].unique())
)

year = st.selectbox(
    "Select Manufacturing Year",
    sorted(df["year"].unique(), reverse=True)
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=50000,
    step=1000
)

# Predict Button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "name": [car_name],
        "company": [company],
        "year": [year],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type]
    })

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Car Price: ₹ {prediction[0]:,.2f}")