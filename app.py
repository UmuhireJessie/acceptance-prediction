import streamlit as st
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("model/driver_acceptance_model.h5")


# Load the scaler from the file
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


st.set_page_config(
    page_title="Driver Acceptance Prediction App",
    page_icon="images/driver.jpeg"
)

st.title("🚘 Acceptance Prediction App 🚘")

st.markdown("<div style='padding: 30px'></div>", unsafe_allow_html=True)
st.subheader("Welcome to My Trained Driver Acceptance Prediction Model")
st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)
st.markdown("""Test if you will be accepted by the driver for sharing the ride and save wasting your time! Enter the basic information about your ride and
                    get the response from the model""")
st.markdown("""
    The model utilizes the predetrained model; Resnet model, to perform feature
    extraction and uses the features extracted to build it. While the user can input the information (data)
    the model but we have limited the model for particular information such as number of passengers, location, distance to be covered, etc, due limited data constraints. To generate predictions, the users must input those infromation, and click the
    predict button.
      
""")

st.markdown("<div style='padding: 50px'></div>", unsafe_allow_html=True)
st.subheader("Enter the following data: ")
st.markdown("<div style='padding: 10px'></div>", unsafe_allow_html=True)

# Input fields
passenger_count = st.number_input("Enter the number of passengers", min_value=1, step=1, value=1)
pickup_longitude = st.number_input("Enter pickup longitude")
pickup_latitude = st.number_input("Enter pickup latitude")
dropoff_longitude = st.number_input("Enter dropoff longitude")
dropoff_latitude = st.number_input("Enter dropoff latitude")
distance_km = st.number_input("Enter distance in kilometers")

# Separate input fields for date and time components
pickup_date = st.date_input("Enter pickup date", datetime.date.today())
pickup_time = st.time_input("Enter pickup time", step=300)

# Combine date and time components
pickup_datetime = datetime.datetime.combine(pickup_date, pickup_time)

# Extract features from pickup_datetime
hour_of_day = pickup_datetime.hour
day_of_week = pickup_datetime.weekday()

# Create a new feature for time of day (morning, afternoon, evening)
def time_of_day_func(hour):
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'

time_of_day = time_of_day_func(hour_of_day)

st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)

# Predict button
predict = st.button("Predict")

if predict:
    # Prepare input features
    input_data = {
        'passenger_count': passenger_count,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'distance_km': distance_km,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'time_of_day': time_of_day
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Standardize input features
    input_features_std = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_features_std)

    # Display prediction result
    if prediction[0] > 0.75:
        st.subheader("The driver is likely to accept the ride.")
    else:
        st.subheader("The driver may not accept the ride.")

    