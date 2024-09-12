
import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained models and encoders from the pickle file
with open('Fire_prediction.pkl', 'rb') as f:
    rf_gas, rf_fire, le_gas, le_fire = pickle.load(f)

# Load an image to use in the app
image = Image.open("fire_safety.jpg")  # Replace with your own image file

# Set the title and display the image
st.title('🔥 Gas and Fire Prediction App 🔥')
st.image(image, caption='Stay safe by predicting potential fire hazards!', use_column_width=True)

# Create a sidebar for user inputs
st.sidebar.header('Input Values')
st.sidebar.write('Enter the parameters below to predict gas type and fire state:')

# Input fields in the sidebar
gas_value = st.sidebar.number_input('Gas Value', min_value=0, value=500, step=1)
gas_increase_rate = st.sidebar.number_input('Gas Increase Rate', min_value=0, value=5, step=1)
temperature = st.sidebar.number_input('Temperature (°C)', min_value=0, value=30, step=1)

# Add some styling for the prediction button
st.sidebar.markdown(
    """
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True
)

# Predict button
if st.sidebar.button('Predict'):
    # Prepare the input data
    input_data = np.array([[gas_value, gas_increase_rate, temperature]])

    # Predict gas type and fire state
    gas_prediction = le_gas.inverse_transform(rf_gas.predict(input_data))[0]
    fire_prediction = le_fire.inverse_transform(rf_fire.predict(input_data))[0]

    # Display the results with custom formatting
    st.subheader('🔍 Prediction Results')
    
    result_col1, result_col2 = st.columns(2)
    
    result_col1.metric("Predicted Gas Type", gas_prediction)
    result_col2.metric("Predicted Fire State", fire_prediction)
    
    # Adding a safety tip section
    st.markdown("""
    ---
    ### Safety Tip 🛑:
    - Always keep an eye on gas readings.
    - If the fire state is predicted to be 'Hazardous', take immediate action.
    """)

# Footer with additional information
st.markdown("""
---
Made with ❤️ using Streamlit. Stay safe!
""")

