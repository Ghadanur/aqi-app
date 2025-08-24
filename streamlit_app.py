import streamlit as st
import pickle
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
import joblib

# Configure page
st.set_page_config(page_title="AQI Predictor", page_icon="🌤️", layout="wide")

# Model URLs - UPDATE THESE based on what's actually in your release
MODEL_URLS = {
    "current": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_current_model.pkl",
    "24h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_24h_model.pkl",
    "48h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_48h_model.pkl",
    "72h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_72h_model.pkl"
}

@st.cache_resource(show_spinner="Loading prediction models...")
def load_models():
    """Load all AQI prediction models from GitHub Releases"""
    models = {}
    for horizon, url in MODEL_URLS.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            models[horizon] = joblib.load(BytesIO(response.content))
            st.sidebar.success(f"✅ {horizon} model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {horizon} model: {str(e)}")
            return None
    return models

def get_aqi_category(aqi_value):
    """Convert AQI value to category"""
    if aqi_value <= 1:
        return "Good", "🟢", "Air quality is satisfactory"
    elif aqi_value <= 2:
        return "Moderate", "🟡", "Air quality is acceptable" 
    elif aqi_value <= 3:
        return "Unhealthy for Sensitive", "🟠", "Sensitive groups may experience minor issues"
    elif aqi_value <= 4:
        return "Unhealthy", "🔴", "Everyone may experience health effects"
    elif aqi_value <= 5:
        return "Very Unhealthy", "🟣", "Health alert: serious health effects"
    else:
        return "Hazardous", "⚫", "Emergency conditions: everyone affected"

def create_feature_vector(input_data):
    """Create feature vector from user input"""
    # This is a simplified version - you'll need to adapt this to match your actual model features
    features = {
        'aqi': input_data.get('aqi', 0),
        'pm2_5': input_data.get('pm2_5', 0),
        'pm10': input_data.get('pm10', 0),
        'co': input_data.get('co', 0),
        'no2': input_data.get('no2', 0),
        'o3': input_data.get('o3', 0),
        'so2': input_data.get('so2', 0),
        'temperature': input_data.get('temperature', 25),
        'humidity': input_data.get('humidity', 50),
        'wind_speed': input_data.get('wind_speed', 5),
        'pressure': input_data.get('pressure', 1013),
        'uv_index': input_data.get('uv_index', 5),
    }
    return pd.DataFrame([features])

def main():
    st.title("🌤️ Real-Time AQI Predictor")
    st.markdown("Predict AQI levels using machine learning models")
    
    # Load models
    with st.spinner("Loading prediction models..."):
        models = load_models()
    
    if not models:
        st.error("""
        ❌ Could not load models. Please check:
        
        1. The model files exist in the GitHub release
        2. The URLs in the code are correct
        3. The files are .pkl format and accessible
        """)
        
        # Debug: Show what's in the release
        with st.expander("🔍 Debug Release Contents"):
            try:
                api_url = "https://api.github.com/repos/Ghadanur/aqi-predictor/releases/tags/models-183-1"
                response = requests.get(api_url)
                release_data = response.json()
                assets = release_data.get('assets', [])
                
                st.write("Files in release:")
                for asset in assets:
                    st.write(f"- {asset['name']} ({(asset['size'] / 1024 / 1024):.2f} MB)")
                    
                    # Try to construct possible URLs
                    if '.pkl' in asset['name']:
                        possible_url = f"https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/{asset['name']}"
                        st.write(f"  Possible URL: `{possible_url}`")
            except Exception as e:
                st.write(f"Error checking release: {e}")
        
        return
    
    st.success(f"✅ Successfully loaded {len(models)} models!")
    
    # User input section
    st.header("📊 Enter Current Environmental Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        aqi = st.slider("Current AQI", 0.0, 6.0, 2.5, 0.1)
        pm2_5 = st.slider("PM2.5 (μg/m³)", 0.0, 500.0, 25.0, 1.0)
        pm10 = st.slider("PM10 (μg/m³)", 0.0, 500.0, 45.0, 1.0)
        co = st.slider("CO (ppm)", 0.0, 10.0, 0.8, 0.1)
    
    with col2:
        so2 = st.slider("SO₂ (ppb)", 0.0, 100.0, 5.0, 1.0)
        no2 = st.slider("NO₂ (ppb)", 0.0, 100.0, 20.0, 1.0)
        o3 = st.slider("O₃ (ppb)", 0.0, 200.0, 80.0, 1.0)
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    
    # Create input data
    input_data = {
        'aqi': aqi,
        'pm2_5': pm2_5,
        'pm10': pm10,
        'co': co,
        'so2': so2,
        'no2': no2,
        'o3': o3,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': 5.0,
        'pressure': 1013.0,
        'uv_index': 5.0
    }
    
    # Make predictions
    if st.button("🚀 Predict AQI", type="primary"):
        with st.spinner("Making predictions..."):
            try:
                features = create_feature_vector(input_data)
                
                predictions = {}
                for horizon, model in models.items():
                    try:
                        pred = model.predict(features)[0]
                        predictions[horizon] = pred
                    except Exception as e:
                        st.error(f"Prediction failed for {horizon}: {str(e)}")
                
                # Display results
                st.header("📈 Prediction Results")
                
                cols = st.columns(4)
                horizons = ['current', '24h', '48h', '72h']
                titles = ['Current', '24H', '48H', '72H']
                
                for i, (horizon, title) in enumerate(zip(horizons, titles)):
                    with cols[i]:
                        if horizon in predictions:
                            aqi_val = predictions[horizon]
                            category, emoji, description = get_aqi_category(aqi_val)
                            
                            st.metric(
                                label=f"{emoji} {title} Forecast",
                                value=f"{aqi_val:.2f}",
                                help=description
                            )
                            st.caption(f"**{category}**")
                        else:
                            st.error(f"No prediction for {title}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
