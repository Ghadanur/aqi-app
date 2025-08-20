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
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TRAINING_REPO = "YOUR_USERNAME/aqi-predictor"  # Replace with your training repo
MODELS_DIR = Path("./models")

# Page configuration
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_latest_release_info():
    """Get information about the latest model release"""
    try:
        api_url = f"https://api.github.com/repos/{TRAINING_REPO}/releases/latest"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Could not fetch release info: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error fetching release info: {str(e)}")
        return None

@st.cache_resource
def download_and_load_models():
    """Download models from GitHub releases and load them"""
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Get latest release info
    release_info = get_latest_release_info()
    
    if not release_info:
        st.error("Could not get latest release information")
        return None, None
    
    # Check if we already have this version
    metadata_file = MODELS_DIR / "metadata.json"
    current_version = None
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                current_version = metadata.get('workflow_run')
        except:
            pass
    
    release_tag = release_info['tag_name']
    
    # Extract run number from tag (format: models-{run_number}-{attempt})
    try:
        release_run = release_tag.split('-')[1]
    except:
        release_run = None
    
    # Download if we don't have this version
    if current_version != release_run:
        with st.spinner(f"Downloading latest models (Release: {release_tag})..."):
            
            # Download all .pkl and .json files from the release
            for asset in release_info.get('assets', []):
                if asset['name'].endswith(('.pkl', '.json')):
                    file_path = MODELS_DIR / asset['name']
                    
                    try:
                        file_response = requests.get(asset['browser_download_url'], timeout=30)
                        file_response.raise_for_status()
                        
                        with open(file_path, 'wb') as f:
                            f.write(file_response.content)
                        
                        logger.info(f"Downloaded: {asset['name']}")
                    except Exception as e:
                        st.error(f"Failed to download {asset['name']}: {str(e)}")
                        return None, None
    
    # Load the predictor classes (you'll need to adapt this based on your model structure)
    try:
        # Import your predictor classes here
        # This is a placeholder - you'll need to implement based on your actual model structure
        predictor = load_predictor_from_files(MODELS_DIR)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        return predictor, metadata
        
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None

def load_predictor_from_files(models_dir):
    """Load predictor from downloaded model files"""
    # This function needs to be implemented based on how your models are saved
    # Example implementation:
    
    class DownloadedPredictor:
        def __init__(self, models_dir):
            self.models_dir = Path(models_dir)
            self.models = {}
            
            # Load all .pkl files
            for pkl_file in self.models_dir.glob("*.pkl"):
                with open(pkl_file, 'rb') as f:
                    self.models[pkl_file.stem] = pickle.load(f)
        
        def predict_from_current_data(self, current_readings):
            # Implement your prediction logic here
            # This is a placeholder
            return {
                '1_hour': {
                    'value': 2.5,
                    'confidence': 'High',
                    'timestamp': (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                    'horizon': '1 hour'
                }
            }
        
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
                return "Hazardous", "🔴", "Emergency conditions"
    
    return DownloadedPredictor(models_dir)

# Rest of your Streamlit app code remains the same, but replace the load_predictor function:

@st.cache_resource
def load_predictor(use_production=True):
    """Load the AQI predictor with caching"""
    try:
        predictor, metadata = download_and_load_models()
        if predictor is None:
            return None, "Failed to download or load models"
        return predictor, None
    except Exception as e:
        return None, str(e)

# Add model info display in sidebar
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

# Add this to your main function before the existing content
def main():
    """Main Streamlit application"""
    
    # Title and header
    st.title("🌤️ AQI Forecast Dashboard")
    st.markdown("Real-time Air Quality Index predictions using machine learning")
    
    # Display model information
    display_model_info()
    
    # Rest of your existing main() function code...
