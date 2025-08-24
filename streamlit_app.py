import streamlit as st
import pickle
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
import re

# Configure page
st.set_page_config(page_title="AQI Predictor", page_icon="🌤️", layout="wide")

def discover_model_files():
    """Automatically discover available model files in the release"""
    try:
        # Get release information
        api_url = "https://api.github.com/repos/Ghadanur/aqi-predictor/releases/tags/models-183-1"
        response = requests.get(api_url)
        release_data = response.json()
        assets = release_data.get('assets', [])
        
        model_files = {}
        patterns = {
            'current': r'(current|aqi_current|model_current)\.pkl$',
            '24h': r'(24h|aqi_24h|model_24h)\.pkl$', 
            '48h': r'(48h|aqi_48h|model_48h)\.pkl$',
            '72h': r'(72h|aqi_72h|model_72h)\.pkl$'
        }
        
        for asset in assets:
            asset_name = asset['name']
            download_url = asset['browser_download_url']
            
            for horizon, pattern in patterns.items():
                if re.search(pattern, asset_name, re.IGNORECASE):
                    model_files[horizon] = download_url
                    break
        
        return model_files
        
    except Exception as e:
        st.error(f"Error discovering files: {e}")
        return {}

def load_model_from_url(url):
    """Load a single model from URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pickle.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load model from {url}: {e}")
        return None

@st.cache_resource(show_spinner="Discovering and loading models...")
def load_models():
    """Load all available models with automatic discovery"""
    model_urls = discover_model_files()
    
    if not model_urls:
        st.error("❌ No model files found in the release!")
        return None
    
    st.sidebar.write("📦 Found model files:")
    for horizon, url in model_urls.items():
        st.sidebar.write(f"- {horizon}: {url.split('/')[-1]}")
    
    models = {}
    for horizon, url in model_urls.items():
        with st.spinner(f"Loading {horizon} model..."):
            model = load_model_from_url(url)
            if model:
                models[horizon] = model
                st.sidebar.success(f"✅ {horizon} model loaded")
            else:
                st.sidebar.error(f"❌ Failed to load {horizon} model")
    
    return models if models else None

# Fallback: Manual URL configuration if automatic discovery fails
FALLBACK_URLS = {
    "current": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_current_model.pkl",
    "24h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_24h_model.pkl",
    "48h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_48h_model.pkl",
    "72h": "https://github.com/Ghadanur/aqi-predictor/releases/download/models-183-1/src/models/aqi_72h_model.pkl"
}

def main():
    st.title("🌤️ Real-Time AQI Predictor")
    st.markdown("Predict AQI levels using machine learning models trained on environmental data")
    
    # Try automatic discovery first
    models = load_models()
    
    # If automatic discovery fails, try fallback URLs
    if not models:
        st.warning("⚠️ Trying fallback URLs...")
        models = {}
        for horizon, url in FALLBACK_URLS.items():
            with st.spinner(f"Trying fallback for {horizon}..."):
                model = load_model_from_url(url)
                if model:
                    models[horizon] = model
    
    if not models:
        st.error("""
        ❌ Could not load any models. Possible reasons:
        
        1. **Model files don't exist** in the release
        2. **File names are different** than expected
        3. **Files are in a subdirectory**
        
        Please check:
        - Your GitHub Actions workflow is saving files correctly
        - The files are actually uploaded to the release
        - The file names match what the app expects
        """)
        
        # Show debug information
        with st.expander("🔍 Debug Information"):
            st.write("Checking release contents...")
            try:
                api_url = "https://api.github.com/repos/Ghadanur/aqi-predictor/releases/tags/models-183-1"
                response = requests.get(api_url)
                release_data = response.json()
                assets = release_data.get('assets', [])
                
                st.write("Files in release:")
                for asset in assets:
                    st.write(f"- {asset['name']} ({(asset['size'] / 1024 / 1024):.2f} MB)")
            except Exception as e:
                st.write(f"Error checking release: {e}")
        
        return
    
    # Rest of your app code...
    st.success(f"✅ Successfully loaded {len(models)} models!")
    
    # Display model info
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"**Release:** models-183-1\n"
                   f"**Commit:** 740b83a\n"
                   f"**Models loaded:** {', '.join(models.keys())}")

if __name__ == "__main__":
    main()
