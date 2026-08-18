import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# App title
st.title('🏡 California House Price Predictor')
st.markdown('Enter the details below to predict the median house value.')

# Input fields
st.sidebar.header('House Features')

longitude = st.sidebar.slider('Longitude', -124.0, -114.0, -119.0)
latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 37.0)
housing_median_age = st.sidebar.slider('Housing Median Age', 1, 52, 20)
total_rooms = st.sidebar.slider('Total Rooms', 100, 10000, 2000)
total_bedrooms = st.sidebar.slider('Total Bedrooms', 50, 2000, 400)
population = st.sidebar.slider('Population', 50, 5000, 1000)
households = st.sidebar.slider('Households', 50, 2000, 400)
median_income = st.sidebar.slider('Median Income (tens of thousands)', 0.5, 15.0, 4.0)
ocean_proximity = st.sidebar.selectbox('Ocean Proximity', 
    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Feature engineering (same as training!)
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

# One-hot encode ocean_proximity
ocean_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_encoded = [1 if ocean_proximity == cat else 0 for cat in ocean_categories]

# Combine all features
features = np.array([[
    longitude, latitude, housing_median_age,
    total_rooms, total_bedrooms, population, households,
    median_income, rooms_per_household, bedrooms_per_room,
    population_per_household, *ocean_encoded
]])

# Scale and predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]

# Display result
st.markdown('---')
st.subheader('Predicted House Value:')
st.metric(label='', value=f'${prediction:,.0f}')

st.markdown('---')
st.markdown('**Model:** XGBoost | **R²:** 0.82 | **MAE:** $28,495')