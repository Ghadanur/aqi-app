import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import json
import logging

# Add your src directory to path (adjust path as needed)
sys.path.append('src')  # Adjust this path to match your project structure

try:
    from models.train import AQIForecastTrainer, RealTimeAQIPredictor
    from features.process import AQI3DayForecastProcessor
except ImportError:
    st.error("❌ Could not import your AQI modules. Please check the path in sys.path.append()")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AQI Forecasting System",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-high { border-left-color: #d32f2f; }
    .prediction-medium { border-left-color: #f57c00; }
    .prediction-low { border-left-color: #388e3c; }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None

def get_aqi_color(aqi_value):
    """Get color based on AQI value"""
    if aqi_value is None:
        return "#gray"
    elif aqi_value <= 1:
        return "#00e400"  # Good
    elif aqi_value <= 2:
        return "#ffff00"  # Moderate
    elif aqi_value <= 3:
        return "#ff7e00"  # Unhealthy for sensitive groups
    elif aqi_value <= 4:
        return "#ff0000"  # Unhealthy
    elif aqi_value <= 5:
        return "#8f3f97"  # Very unhealthy
    else:
        return "#7e0023"  # Hazardous

def get_aqi_label(aqi_value):
    """Get AQI category label"""
    if aqi_value is None:
        return "Unknown"
    elif aqi_value <= 1:
        return "Good"
    elif aqi_value <= 2:
        return "Moderate"
    elif aqi_value <= 3:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 4:
        return "Unhealthy"
    elif aqi_value <= 5:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def training_tab():
    """Training tab interface"""
    st.header("🔬 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Train AQI Forecasting Models
        This will train models for predicting AQI at different time horizons:
        - **Current + 1h**: Immediate forecast
        - **24h**: Tomorrow's AQI
        - **48h**: Day after tomorrow
        - **72h**: 3-day forecast
        """)
    
    with col2:
        st.info("""
        **Requirements:**
        - Historical AQI data
        - Weather data
        - At least 100 data points
        """)
    
    # Training controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Training", type="primary"):
            train_models()
    
    with col2:
        if st.button("📊 Load Existing Models"):
            load_existing_models()
    
    with col3:
        if st.button("🗑️ Clear Models"):
            clear_models()
    
    # Display training results
    if st.session_state.training_results:
        display_training_results()

def train_models():
    """Train the AQI forecasting models"""
    with st.spinner("Training models... This may take a few minutes."):
        try:
            # Initialize trainer
            st.session_state.trainer = AQIForecastTrainer()
            
            # Prepare data
            st.info("📊 Preparing data...")
            st.session_state.trainer.prepare_data()
            
            if not st.session_state.trainer.datasets:
                st.error("❌ No valid datasets prepared. Check your data quality.")
                return
            
            # Train models
            st.info("🔧 Training models...")
            results = st.session_state.trainer.train_models()
            
            # Save models
            st.info("💾 Saving models...")
            st.session_state.trainer.save_models()
            
            st.session_state.training_results = results
            st.session_state.models_trained = True
            
            st.success("✅ Training completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            logger.error(f"Training error: {str(e)}")

def load_existing_models():
    """Load existing trained models"""
    try:
        if not os.path.exists('models'):
            st.warning("📁 No models directory found. Please train models first.")
            return
        
        model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
        
        if not model_files:
            st.warning("📁 No trained models found. Please train models first.")
            return
        
        st.session_state.predictor = RealTimeAQIPredictor(
            models_dir='models', 
            trainer_instance=st.session_state.trainer
        )
        st.session_state.models_trained = True
        st.success(f"✅ Loaded {len(model_files)} trained models!")
        
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")

def clear_models():
    """Clear trained models"""
    st.session_state.trainer = None
    st.session_state.predictor = None
    st.session_state.models_trained = False
    st.session_state.training_results = None
    st.info("🧹 Models cleared from session.")

def display_training_results():
    """Display training results"""
    st.subheader("📈 Training Results")
    
    results_data = []
    for horizon, result in st.session_state.training_results.items():
        if 'test_mae' in result and result['test_mae'] is not None:
            results_data.append({
                'Horizon': horizon.replace('aqi_', '').replace('current', '1h'),
                'MAE': round(result['test_mae'], 3),
                'Status': '✅ Success'
            })
        else:
            results_data.append({
                'Horizon': horizon.replace('aqi_', '').replace('current', '1h'),
                'MAE': 'N/A',
                'Status': '❌ Failed'
            })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Feature importance for successful models
    for horizon, result in st.session_state.training_results.items():
        if 'feature_importance' in result and not result['feature_importance'].empty:
            with st.expander(f"🔍 Feature Importance - {horizon}"):
                fig = px.bar(
                    result['feature_importance'].head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 10 Features for {horizon}"
                )
                st.plotly_chart(fig, use_container_width=True)

def prediction_tab():
    """Real-time prediction tab"""
    st.header("🔮 Real-Time AQI Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train or load models first in the Training tab.")
        return
    
    # Initialize predictor if not already done
    if st.session_state.predictor is None:
        try:
            st.session_state.predictor = RealTimeAQIPredictor(
                models_dir='models',
                trainer_instance=st.session_state.trainer
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize predictor: {str(e)}")
            return
    
    # Input form
    st.subheader("📊 Current Sensor Readings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌬️ Air Quality**")
        current_aqi = st.number_input("Current AQI", min_value=0.0, max_value=6.0, value=2.5, step=0.1)
        pm2_5 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, value=25.0, step=1.0)
        pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, value=45.0, step=1.0)
        co = st.number_input("CO (ppm)", min_value=0.0, value=0.8, step=0.1)
    
    with col2:
        st.markdown("**🏭 Gas Pollutants**")
        no2 = st.number_input("NO₂ (ppb)", min_value=0.0, value=20.0, step=1.0)
        o3 = st.number_input("O₃ (ppb)", min_value=0.0, value=80.0, step=1.0)
        so2 = st.number_input("SO₂ (ppb)", min_value=0.0, value=5.0, step=1.0)
    
    with col3:
        st.markdown("**🌡️ Weather**")
        temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=28.5, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=5.0, step=0.1)
        pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.25, step=0.01)
        uv_index = st.number_input("UV Index", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    
    # Prediction button
    if st.button("🔮 Generate Predictions", type="primary"):
        make_predictions({
            'aqi': current_aqi,
            'pm2_5': pm2_5,
            'pm10': pm10,
            'co': co,
            'no2': no2,
            'o3': o3,
            'so2': so2,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'uv_index': uv_index,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

def make_predictions(current_data):
    """Make and display predictions"""
    try:
        with st.spinner("🔮 Making predictions..."):
            predictions = st.session_state.predictor.predict_from_current_data(current_data)
        
        st.success("✅ Predictions generated successfully!")
        display_predictions(predictions, current_data)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")

def display_predictions(predictions, current_data):
    """Display prediction results"""
    st.subheader("📊 AQI Forecasts")
    
    # Current AQI display
    current_aqi = current_data['aqi']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current AQI</h4>
            <h2 style="color: {get_aqi_color(current_aqi)};">{current_aqi:.1f}</h2>
            <p>{get_aqi_label(current_aqi)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Predictions display
    horizons = ['aqi_24h', 'aqi_48h', 'aqi_72h']
    horizon_labels = ['24h Forecast', '48h Forecast', '72h Forecast']
    
    for i, (horizon, label) in enumerate(zip(horizons, horizon_labels)):
        if horizon in predictions and predictions[horizon]['value'] is not None:
            pred_value = predictions[horizon]['value']
            confidence = predictions[horizon]['confidence']
            
            # Determine card style based on confidence
            card_class = f"prediction-{confidence.lower()}" if confidence.lower() in ['high', 'medium', 'low'] else ""
            
            with [col2, col3, col4][i]:
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h4>{label}</h4>
                    <h2 style="color: {get_aqi_color(pred_value)};">{pred_value:.1f}</h2>
                    <p>{get_aqi_label(pred_value)}</p>
                    <small>Confidence: {confidence}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Trend visualization
    create_trend_chart(current_data, predictions)
    
    # Detailed prediction table
    create_prediction_table(predictions)

def create_trend_chart(current_data, predictions):
    """Create trend visualization"""
    st.subheader("📈 AQI Trend Forecast")
    
    # Prepare data for plotting
    timestamps = [datetime.now()]
    values = [current_data['aqi']]
    
    for horizon in ['aqi_24h', 'aqi_48h', 'aqi_72h']:
        if horizon in predictions and predictions[horizon]['value'] is not None:
            if 'timestamp' in predictions[horizon]:
                ts = pd.to_datetime(predictions[horizon]['timestamp'])
            else:
                hours_ahead = int(horizon.split('_')[1].replace('h', ''))
                ts = datetime.now() + timedelta(hours=hours_ahead)
            
            timestamps.append(ts)
            values.append(predictions[horizon]['value'])
    
    # Create plotly chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name='AQI Forecast',
        line=dict(width=3),
        marker=dict(size=10)
    ))
    
    # Add color zones
    fig.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=2, line_dash="dash", line_color="yellow", annotation_text="Moderate")
    fig.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Unhealthy")
    fig.add_hline(y=4, line_dash="dash", line_color="red", annotation_text="Very Unhealthy")
    
    fig.update_layout(
        title="AQI Forecast Trend",
        xaxis_title="Time",
        yaxis_title="AQI Value",
        yaxis=dict(range=[0, max(6, max(values) + 0.5)]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_prediction_table(predictions):
    """Create detailed predictions table"""
    st.subheader("📋 Detailed Predictions")
    
    table_data = []
    for horizon, pred in predictions.items():
        if pred['value'] is not None:
            table_data.append({
                'Time Horizon': pred['horizon'],
                'Predicted AQI': f"{pred['value']:.2f}",
                'Category': get_aqi_label(pred['value']),
                'Confidence': pred['confidence'],
                'Timestamp': pred.get('timestamp', 'N/A')
            })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

def main():
    """Main application"""
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("🌬️ AQI Forecasting System")
    st.sidebar.markdown("---")
    
    # Model status
    if st.session_state.models_trained:
        st.sidebar.success("✅ Models Ready")
    else:
        st.sidebar.warning("⚠️ Models Not Loaded")
    
    st.sidebar.markdown("---")
    
    # Navigation
    tab = st.sidebar.radio(
        "Navigation",
        ["🔬 Training", "🔮 Prediction", "📊 About"]
    )
    
    # Main content
    if tab == "🔬 Training":
        training_tab()
    elif tab == "🔮 Prediction":
        prediction_tab()
    else:
        about_tab()

def about_tab():
    """About tab"""
    st.header("📊 About AQI Forecasting System")
    
    st.markdown("""
    ### 🎯 Overview
    This system provides comprehensive Air Quality Index (AQI) forecasting using machine learning models.
    
    ### 🔬 Features
    - **Multi-horizon Prediction**: 1h, 24h, 48h, and 72h forecasts
    - **Multiple Pollutants**: PM2.5, PM10, CO, NO₂, O₃, SO₂
    - **Weather Integration**: Temperature, humidity, wind, pressure
    - **Real-time Prediction**: Input current readings for instant forecasts
    - **Model Training**: Train on your own historical data
    
    ### 📈 Model Architecture
    - **Base Model**: LightGBM with time-series validation
    - **Fallback**: Linear Regression for robustness
    - **Features**: 40+ engineered features including lags and interactions
    - **Validation**: Time-series split to prevent data leakage
    
    ### 🎨 AQI Categories
    - **0-1**: Good (Green)
    - **1-2**: Moderate (Yellow)
    - **2-3**: Unhealthy for Sensitive Groups (Orange)
    - **3-4**: Unhealthy (Red)
    - **4-5**: Very Unhealthy (Purple)
    - **5+**: Hazardous (Maroon)
    """)
    
    st.info("💡 **Tip**: For best results, ensure your training data covers diverse weather conditions and seasonal patterns.")

if __name__ == "__main__":
    main()
