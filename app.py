import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
import math  


model = pk.load(open('model.pkl', 'rb'))

st.header('Car Price Prediction ML Model')


cars_data = pd.read_csv(r'CardataSet.csv')


def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()


cars_data['name'] = cars_data['name'].apply(get_brand_name)

# User inputs
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 0, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (kmpl)', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power (bhp)', 0, 200)
seats = st.slider('No of Seats', 2, 10)

if st.button("Predict"):
   
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    
    input_data_model = pd.get_dummies(input_data_model, drop_first=True)

    
    model_columns = model.feature_names_in_  

    
    for col in model_columns:
        if col not in input_data_model.columns:
            input_data_model[col] = 0 

    input_data_model = input_data_model[model_columns]  

    
    car_price = model.predict(input_data_model)

    
    st.markdown('Car Price is going to be ' + str(math.floor(car_price[0])))