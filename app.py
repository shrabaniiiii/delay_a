mport streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Make sure 'logi.sav' is in the same directory as this app.py file
logi = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input features (based on X.columns)
# From previous output: Index(['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
#        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
#        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
#        'Warehouse_Processing_Time'], dtype='object')

delivery_distance = st.slider('Delivery Distance (km)', 1.0, 50.0, 25.0)
traffic_congestion = st.selectbox('Traffic Congestion (1-5, 5=High)', [1, 2, 3, 4, 5])
weather_condition = st.selectbox('Weather Condition (1-5, 5=Bad)', [1, 2, 3, 4, 5])
delivery_slot = st.selectbox('Delivery Slot (1-3)', [1, 2, 3])
driver_experience = st.slider('Driver Experience (years)', 0, 20, 5)
num_stops = st.slider('Number of Stops', 1, 10, 3)
vehicle_age = st.slider('Vehicle Age (years)', 0, 15, 5)
road_condition_score = st.selectbox('Road Condition Score (1-5, 5=Good)', [1, 2, 3, 4, 5])
package_weight = st.slider('Package Weight (kg)', 0.1, 150.0, 50.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 5.0, 25.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 1, 120, 60)

# Create a DataFrame for the input
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = logi.predict(input_data)
    prediction_proba = logi.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f'Prediction: \n *Delivery Delay is Likely!*')
    else:
        st.success(f'Prediction: \n *No Delivery Delay is Expected.*')
    
    st.write(f'Probability of Delay: {prediction_proba[0][1]:.2f}')
    st.write(f'Probability of No Delay: {prediction_proba[0][0]:.2f}')
