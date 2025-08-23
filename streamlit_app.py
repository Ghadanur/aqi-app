import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import logging
import requests
import os
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Update this with your actual GitHub repository
TRAINING_REPO = "ghadanur/aqi-predictor"  # Update with your actual repo
MODELS_DIR = Path("./models")

# Page configuration
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock predictor class for demo purposes
class DemoPredictor:
    def __init__(self):
        self.model_version = "1.0.0"
    
    def predict_from_current_data(self, current_readings):
        """Generate demo predictions"""
        base_aqi = np.random.uniform(1.5, 3.5)
        
        predictions = {}
        for i, hours in enumerate([1, 3, 6, 12, 24]):
            # Add some variation
            variation = np.random.uniform(-0.3, 0.3)
            predicted_aqi = max(1.0, base_aqi + variation + (i * 0.1))
            
            predictions[f'{hours}_hour'] = {
                'value': round(predicted_aqi, 2),
                'confidence': np.random.choice(['High', 'Medium', 'High'], p=[0.6, 0.2, 0.2]),
                'timestamp': (datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S'),
                'horizon': f'{hours} hour{"s" if hours > 1 else ""}'
            }
        
        return predictions
    
    def predict_with_buffer(self, current_readings):
        return self.predict_from_current_data(current_readings)
    
    def get_aqi_category(self, aqi_value):
        if aqi_value <= 1:
            return "Good", "🟢", "Air quality is satisfactory"
        elif aqi_value <= 2:
            return "Moderate", "🟡", "Air quality is acceptable"
        elif aqi_value <= 3:
            return "Unhealthy for Sensitive", "🟠", "Sensitive groups may experience issues"
        elif aqi_value <= 4:
            return "Unhealthy", "🔴", "Everyone may experience health effects"
        elif aqi_value <= 5:
            return "Very Unhealthy", "🟣", "Health alert: serious health effects"
        else:
            return "Hazardous", "⚫", "Emergency conditions"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_latest_release_info():
    """Get information about the latest model release"""
    try:
        if TRAINING_REPO == "ghadanur/aqi-predictor":  # Default placeholder
            return None  # Skip GitHub API call for demo
            
        api_url = f"https://api.github.com/repos/{TRAINING_REPO}/releases/latest"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Could not fetch release info: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Error fetching release info: {str(e)}")
        return None

@st.cache_resource
def load_predictor(use_demo=True):
    """Load the AQI predictor with caching"""
    try:
        if use_demo or TRAINING_REPO == "ghadanur/aqi-predictor":
            # Use demo predictor
            predictor = DemoPredictor()
            return predictor, None
        
        # Try to load from GitHub releases
        predictor, metadata = download_and_load_models()
        if predictor is None:
            # Fallback to demo
            predictor = DemoPredictor()
            return predictor, "Using demo predictor - could not load from GitHub"
        
        return predictor, None
    except Exception as e:
        # Fallback to demo
        predictor = DemoPredictor()
        return predictor, f"Using demo predictor - Error: {str(e)}"

def download_and_load_models():
    """Download models from GitHub releases and load them"""
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Get latest release info
    release_info = get_latest_release_info()
    
    if not release_info:
        return None, None
    
    # Implementation for actual model loading would go here
    # For now, return None to use demo predictor
    return None, None

def generate_sample_data():
    """Generate sample historical data for visualization"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate realistic AQI data with some pattern
    base_trend = np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7)) * 0.5  # Weekly pattern
    daily_pattern = np.sin(np.arange(len(dates)) * 2 * np.pi / 24) * 0.3  # Daily pattern
    noise = np.random.normal(0, 0.2, len(dates))
    
    aqi_values = 2.5 + base_trend + daily_pattern + noise
    aqi_values = np.clip(aqi_values, 1.0, 5.0)
    
    return pd.DataFrame({
        'timestamp': dates,
        'aqi': aqi_values,
        'pm25': aqi_values * 25 + np.random.normal(0, 5, len(dates)),
        'pm10': aqi_values * 40 + np.random.normal(0, 8, len(dates)),
        'temperature': 25 + np.random.normal(0, 5, len(dates)),
        'humidity': 60 + np.random.normal(0, 10, len(dates))
    })

def display_model_info():
    """Display information about the loaded models"""
    with st.sidebar:
        st.subheader("🤖 Model Information")
        
        release_info = get_latest_release_info()
        if release_info:
            st.info(f"**Latest Release:** {release_info['tag_name']}")
            st.info(f"**Published:** {release_info['published_at'][:10]}")
            
            # Show release notes if available
            if release_info.get('body'):
                with st.expander("📝 Release Notes"):
                    st.markdown(release_info['body'])
        else:
            st.info("**Demo Mode:** Using sample predictions")
            st.info("**Version:** 1.0.0-demo")
            
        st.markdown("---")
        st.markdown("**Configuration:**")
        st.code(f"Repository: {TRAINING_REPO}")

def create_prediction_chart(predictions, historical_data):
    """Create an interactive prediction chart"""
    fig = go.Figure()
    
    # Add historical data
    recent_data = historical_data.tail(48)  # Last 48 hours
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data['aqi'],
        mode='lines',
        name='Historical AQI',
        line=dict(color='blue', width=2)
    ))
    
    # Add predictions
    pred_times = []
    pred_values = []
    pred_confidences = []
    
    for key, pred in predictions.items():
        pred_time = datetime.strptime(pred['timestamp'], '%Y-%m-%d %H:%M:%S')
        pred_times.append(pred_time)
        pred_values.append(pred['value'])
        pred_confidences.append(pred['confidence'])
    
    fig.add_trace(go.Scatter(
        x=pred_times,
        y=pred_values,
        mode='lines+markers',
        name='Predictions',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Add AQI category zones
    fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=2, line_dash="dot", line_color="yellow", opacity=0.3)
    fig.add_hline(y=3, line_dash="dot", line_color="orange", opacity=0.3)
    fig.add_hline(y=4, line_dash="dot", line_color="red", opacity=0.3)
    fig.add_hline(y=5, line_dash="dot", line_color="purple", opacity=0.3)
    
    fig.update_layout(
        title="AQI Forecast",
        xaxis_title="Time",
        yaxis_title="AQI Value",
        hovermode='x unified',
        height=500
    )
    
    return fig

def display_current_conditions(current_readings, predictor):
    """Display current air quality conditions"""
    current_aqi = current_readings.get('aqi', 2.5)
    category, emoji, description = predictor.get_aqi_category(current_aqi)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current AQI",
            value=f"{current_aqi:.1f}",
            delta=f"{np.random.uniform(-0.2, 0.2):.1f}"
        )
    
    with col2:
        st.markdown(f"### {emoji} {category}")
        st.caption(description)
    
    with col3:
        st.metric(
            label="PM2.5 (μg/m³)",
            value=f"{current_readings.get('pm25', 65):.0f}",
            delta=f"{np.random.uniform(-5, 5):.0f}"
        )

def display_predictions(predictions, predictor):
    """Display prediction results"""
    st.subheader("🔮 Predictions")
    
    cols = st.columns(len(predictions))
    
    for i, (key, pred) in enumerate(predictions.items()):
        with cols[i]:
            category, emoji, _ = predictor.get_aqi_category(pred['value'])
            
            # Create a colored metric card
            st.markdown(f"""
            <div style="
                padding: 1rem; 
                border-radius: 0.5rem; 
                border-left: 4px solid {'#22c55e' if pred['value'] <= 2 else '#f59e0b' if pred['value'] <= 3 else '#ef4444'};
                background-color: #f9fafb;
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0; color: #1f2937;">{pred['horizon']}</h4>
                <h2 style="margin: 0.5rem 0; color: #1f2937;">{pred['value']} {emoji}</h2>
                <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">{category}</p>
                <p style="margin: 0.25rem 0 0 0; color: #9ca3af; font-size: 0.75rem;">Confidence: {pred['confidence']}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.title("🌤️ AQI Forecast Dashboard")
    st.markdown("Real-time Air Quality Index predictions using machine learning")
    
    # Display model information in sidebar
    display_model_info()
    
    # Load predictor
    with st.spinner("Loading prediction model..."):
        predictor, error_msg = load_predictor()
    
    if error_msg:
        st.warning(error_msg)
    
    if predictor is None:
        st.error("Failed to load prediction model. Please check your configuration.")
        st.stop()
    
    # Sidebar controls
    st.sidebar.subheader("🎛️ Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    
    if st.sidebar.button("🔄 Refresh Data") or auto_refresh:
        st.cache_data.clear()
        st.rerun()
    
    # Generate sample current readings
    current_readings = {
        'aqi': np.random.uniform(1.8, 3.2),
        'pm25': np.random.uniform(40, 80),
        'pm10': np.random.uniform(60, 120),
        'temperature': np.random.uniform(20, 30),
        'humidity': np.random.uniform(50, 70)
    }
    
    # Display current conditions
    st.subheader("📊 Current Conditions")
    display_current_conditions(current_readings, predictor)
    
    # Get predictions
    with st.spinner("Generating predictions..."):
        predictions = predictor.predict_from_current_data(current_readings)
    
    # Display predictions
    display_predictions(predictions, predictor)
    
    # Create and display charts
    st.subheader("📈 Forecast Chart")
    
    # Generate sample historical data
    historical_data = generate_sample_data()
    
    # Create prediction chart
    chart = create_prediction_chart(predictions, historical_data)
    st.plotly_chart(chart, use_container_width=True)
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Environmental Factors")
        env_data = pd.DataFrame({
            'Metric': ['Temperature', 'Humidity', 'Wind Speed'],
            'Value': [f"{current_readings['temperature']:.1f}°C", 
                     f"{current_readings['humidity']:.0f}%", 
                     f"{np.random.uniform(5, 15):.1f} km/h"],
            'Status': ['Normal', 'Normal', 'Normal']
        })
        st.dataframe(env_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("⚠️ Health Recommendations")
        current_aqi = current_readings['aqi']
        if current_aqi <= 2:
            st.success("✅ Air quality is good. Safe for outdoor activities.")
        elif current_aqi <= 3:
            st.warning("⚠️ Moderate air quality. Sensitive individuals should limit outdoor exposure.")
        else:
            st.error("🚨 Unhealthy air quality. Limit outdoor activities and wear masks.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **About**: This dashboard provides AI-powered air quality predictions. "
        "Data is refreshed regularly and predictions are updated in real-time."
    )
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()

if __name__ == "__main__":
    main()
