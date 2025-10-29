import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and data
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv('cleaned_global_water_consumption.csv')

st.title("🌊 Smart Water Consumption Predictor")

st.write("Predict future water usage based on country & year")

# Input fields
country = st.text_input("Enter Country")
year = st.number_input("Enter Year", min_value=2000, max_value=2100, step=1)

if st.button("Predict"):
    try:
        df_country = data[data['Country'].str.lower() == country.lower()]

        if df_country.empty:
            st.error(f"Country '{country}' not found in dataset.")
        else:
            last_row = df_country.iloc[-1]

            features = np.array([[ 
                last_row['Per Capita Water Use (Liters per Day)'],
                last_row['Agricultural Water Use (%)'],
                last_row['Industrial Water Use (%)'],
                last_row['Household Water Use (%)'],
                last_row['Rainfall Impact (Annual Precipitation in mm)'],
                last_row['Groundwater Depletion Rate (%)'],
                year
            ]])

            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]

            st.subheader(f"🚰 Predicted Water Consumption: {prediction:.2f} L/day")

            if prediction > 600:
                st.warning(f"🌊 High demand expected for {country}. Adopt water-saving innovations.")
            elif 400 <= prediction <= 600:
                st.info(f"💧 Moderate usage predicted. Encourage conservation & recycling.")
            else:
                st.success(f"🌿 Efficient usage trend detected. Maintain sustainability efforts.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
