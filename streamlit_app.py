import streamlit as st
import pickle
import requests
import os
from io import BytesIO

# Configuration
MODEL_RELEASE_URL = "https://github.com/Ghadanur/aqi-predictor/releases/latest/download/aqi_current_model.pkl"

@st.cache_resource(show_spinner="Downloading prediction model...")
def load_model_from_url(url):
    """Download and load model from GitHub Releases"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        model = pickle.load(BytesIO(response.content))
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

def main():
    st.title("AQI Prediction App")
    
    # Load model
    model = load_model_from_url(MODEL_RELEASE_URL)
    
    if model:
        # Your prediction code here
        input_data = get_user_input()  # Your function to get input
        prediction = model.predict(input_data)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
