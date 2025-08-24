import streamlit as st
import pickle
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime

# Configure page
st.set_page_config(page_title="AQI Predictor", page_icon="🌤️", layout="wide")

# Model URLs from your GitHub release
MODEL_URLS = {
    "current": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/aqi_current_model.pkl",
    "24h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/aqi_24h_model.pkl",
    "48h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/aqi_48h_model.pkl", 
    "72h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/aqi_72h_model.pkl"
}

@st.cache_resource(show_spinner="Loading prediction models...")
def load_models():
    """Load all AQI prediction models from GitHub Releases"""
    models = {}
    for horizon, url in MODEL_URLS.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            models[horizon] = pickle.load(BytesIO(response.content))
            st.sidebar.success(f"✅ {horizon} model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {horizon} model")
            st.sidebar.error(str(e))
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
    """Create feature vector from user input matching training format"""
    # This should match exactly what your model expects
    # You'll need to adapt this based on your actual feature engineering
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
        'hour_sin': np.sin(2 * np.pi * datetime.now().hour / 24),
        'hour_cos': np.cos(2 * np.pi * datetime.now().hour / 24),
        # Add other features your model expects...
    }
    return pd.DataFrame([features])

def main():
    st.title("🌤️ Real-Time AQI Predictor")
    st.markdown("Predict AQI levels using machine learning models trained on environmental data")
    
    # Load models
    with st.spinner("Loading prediction models from GitHub..."):
        models = load_models()
    
    if not models:
        st.error("Failed to load models. Please check the release availability.")
        return
    
    # Display model info
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"**Release:** models-183-1\n"
                   f"**Commit:** 740b83a\n"
                   f"**Trained:** 11 hours ago")
    
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
        'wind_speed': 5.0,  # Default values
        'pressure': 1013.0,
        'uv_index': 5.0
    }
    
    # Make predictions
    if st.button("🚀 Predict AQI", type="primary"):
        with st.spinner("Making predictions..."):
            try:
                # Create feature vector
                features = create_feature_vector(input_data)
                
                # Make predictions for each horizon
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
    
    # Add information section
    st.markdown("---")
    st.info(
        "**Note:** These predictions are based on machine learning models trained on historical environmental data. "
        "For the most accurate predictions, ensure all input values are as current as possible."
    )

if __name__ == "__main__":
    main()
