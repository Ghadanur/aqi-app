# streamlit_aqi_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import logging
from typing import Dict, List, Optional

# Import your prediction classes (adjust the import path as needed)
try:
    from src.models.train import RealTimeAQIPredictor, ProductionAQIPredictor, AQIForecastTrainer
except ImportError:
    # If direct import fails, you may need to adjust the path
    st.error("Could not import AQI prediction modules. Please check the file paths.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; }
    .status-moderate { color: #ffc107; }
    .status-unhealthy { color: #fd7e14; }
    .status-danger { color: #dc3545; }
    .status-hazardous { color: #6f42c1; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor(use_production=True):
    """Load the AQI predictor with caching"""
    try:
        if use_production:
            predictor = ProductionAQIPredictor(trainer_instance=None)
        else:
            predictor = RealTimeAQIPredictor(trainer_instance=None)
        return predictor, None
    except Exception as e:
        return None, str(e)

def get_aqi_color(aqi_value):
    """Get color based on AQI value"""
    if aqi_value <= 1:
        return "#28a745"  # Green
    elif aqi_value <= 2:
        return "#ffc107"  # Yellow
    elif aqi_value <= 3:
        return "#fd7e14"  # Orange
    elif aqi_value <= 4:
        return "#dc3545"  # Red
    elif aqi_value <= 5:
        return "#6f42c1"  # Purple
    else:
        return "#dc3545"  # Dark Red

def create_aqi_gauge(aqi_value, title):
    """Create a gauge chart for AQI value"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 3.0},
        gauge = {
            'axis': {'range': [None, 6]},
            'bar': {'color': get_aqi_color(aqi_value)},
            'steps': [
                {'range': [0, 1], 'color': "#d4edda"},
                {'range': [1, 2], 'color': "#fff3cd"},
                {'range': [2, 3], 'color': "#fdebd0"},
                {'range': [3, 4], 'color': "#f8d7da"},
                {'range': [4, 5], 'color': "#e2d6f3"},
                {'range': [5, 6], 'color': "#d1ecf1"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_forecast_chart(predictions):
    """Create a forecast chart"""
    times = []
    values = []
    confidences = []
    
    for horizon, pred in predictions.items():
        if pred.get('value') is not None:
            times.append(pred['timestamp'])
            values.append(pred['value'])
            confidences.append(pred['confidence'])
    
    if not times:
        return None
    
    df = pd.DataFrame({
        'Time': pd.to_datetime(times),
        'AQI': values,
        'Confidence': confidences
    })
    
    fig = px.line(df, x='Time', y='AQI', 
                  title='AQI Forecast Timeline',
                  markers=True,
                  color_discrete_sequence=['#1f77b4'])
    
    # Add AQI zones
    fig.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=2, line_dash="dash", line_color="yellow", annotation_text="Moderate") 
    fig.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive")
    fig.add_hline(y=4, line_dash="dash", line_color="red", annotation_text="Unhealthy")
    fig.add_hline(y=5, line_dash="dash", line_color="purple", annotation_text="Very Unhealthy")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="AQI Value",
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.title("🌤️ AQI Forecast Dashboard")
    st.markdown("Real-time Air Quality Index predictions using machine learning")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Sensor Inputs")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method:",
        ["Manual Input", "Upload CSV", "Random Sample"]
    )
    
    current_readings = {}
    
    if input_method == "Manual Input":
        st.sidebar.subheader("Current Readings")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            current_readings['aqi'] = st.number_input("Current AQI", min_value=0.0, max_value=10.0, value=3.2, step=0.1)
            current_readings['pm2_5'] = st.number_input("PM2.5 (μg/m³)", min_value=0.0, value=28.5, step=0.1)
            current_readings['pm10'] = st.number_input("PM10 (μg/m³)", min_value=0.0, value=45.2, step=0.1)
            current_readings['co'] = st.number_input("CO (mg/m³)", min_value=0.0, value=0.9, step=0.01)
            current_readings['no2'] = st.number_input("NO₂ (μg/m³)", min_value=0.0, value=22.1, step=0.1)
            current_readings['o3'] = st.number_input("O₃ (μg/m³)", min_value=0.0, value=85.3, step=0.1)
        
        with col2:
            current_readings['so2'] = st.number_input("SO₂ (μg/m³)", min_value=0.0, value=4.8, step=0.1)
            current_readings['temperature'] = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=29.8, step=0.1)
            current_readings['humidity'] = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=68.5, step=0.1)
            current_readings['wind_speed'] = st.number_input("Wind Speed (m/s)", min_value=0.0, value=6.2, step=0.1)
            current_readings['pressure'] = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1200.0, value=1012.5, step=0.1)
            current_readings['uv_index'] = st.number_input("UV Index", min_value=0.0, max_value=15.0, value=7.1, step=0.1)
    
    elif input_method == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if len(df) > 0:
                    # Use the first row or latest timestamp
                    latest_row = df.iloc[-1]
                    for col in df.columns:
                        if col.lower() in ['aqi', 'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'temperature', 'humidity', 'wind_speed', 'pressure', 'uv_index']:
                            current_readings[col.lower()] = float(latest_row[col])
                    st.sidebar.success(f"Loaded data from CSV ({len(df)} rows)")
                else:
                    st.sidebar.error("CSV file is empty")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {str(e)}")
    
    elif input_method == "Random Sample":
        if st.sidebar.button("Generate Random Sample"):
            current_readings = {
                'aqi': round(2.0 + np.random.random() * 2, 2),
                'pm2_5': round(20 + np.random.random() * 30, 1),
                'pm10': round(35 + np.random.random() * 25, 1),
                'co': round(0.5 + np.random.random() * 1.0, 2),
                'so2': round(3 + np.random.random() * 5, 1),
                'no2': round(15 + np.random.random() * 15, 1),
                'o3': round(70 + np.random.random() * 30, 1),
                'temperature': round(25 + np.random.random() * 10, 1),
                'humidity': round(50 + np.random.random() * 30, 1),
                'wind_speed': round(3 + np.random.random() * 8, 1),
                'pressure': round(1000 + np.random.random() * 30, 1),
                'uv_index': round(3 + np.random.random() * 8, 1),
            }
    
    # Add timestamp
    current_readings['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Model selection
    use_production = st.sidebar.checkbox("Use Production Model (with buffer)", value=True)
    
    # Main content area
    if current_readings and len(current_readings) > 1:  # More than just timestamp
        
        # Load predictor
        with st.spinner("Loading prediction model..."):
            predictor, error = load_predictor(use_production)
        
        if predictor is None:
            st.error(f"Failed to load predictor: {error}")
            st.info("Make sure your models are trained and saved in the 'models' directory")
            st.stop()
        
        # Display current readings
        st.subheader("📈 Current Sensor Readings")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current AQI", f"{current_readings.get('aqi', 0):.2f}", 
                     delta=None, delta_color="inverse")
        with col2:
            st.metric("PM2.5", f"{current_readings.get('pm2_5', 0):.1f} μg/m³")
        with col3:
            st.metric("Temperature", f"{current_readings.get('temperature', 0):.1f}°C")
        with col4:
            st.metric("Humidity", f"{current_readings.get('humidity', 0):.1f}%")
        
        # Make predictions
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Generating AQI forecast..."):
                try:
                    if use_production:
                        predictions = predictor.predict_with_buffer(current_readings)
                    else:
                        predictions = predictor.predict_from_current_data(current_readings)
                    
                    # Display predictions
                    st.subheader("🌤️ AQI Forecast Results")
                    
                    # Create columns for different time horizons
                    cols = st.columns(len([p for p in predictions.values() if p.get('value') is not None]))
                    
                    col_idx = 0
                    for horizon, pred_info in predictions.items():
                        if pred_info.get('value') is not None:
                            with cols[col_idx]:
                                aqi_val = pred_info['value']
                                category, emoji, description = predictor.get_aqi_category(aqi_val)
                                
                                # Create gauge chart
                                gauge_fig = create_aqi_gauge(aqi_val, f"{pred_info['horizon'].upper()}")
                                st.plotly_chart(gauge_fig, use_container_width=True)
                                
                                # Display details
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h4>{emoji} {pred_info['horizon'].upper()}</h4>
                                    <p><strong>AQI:</strong> {aqi_val:.2f}</p>
                                    <p><strong>Category:</strong> {category}</p>
                                    <p><strong>Confidence:</strong> {pred_info['confidence']}</p>
                                    <p><strong>Target Time:</strong> {pred_info['timestamp']}</p>
                                    <p><em>{description}</em></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            col_idx += 1
                    
                    # Create forecast timeline
                    forecast_fig = create_forecast_chart(predictions)
                    if forecast_fig:
                        st.subheader("📊 Forecast Timeline")
                        st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    # Show detailed results table
                    with st.expander("📋 Detailed Results"):
                        results_data = []
                        for horizon, pred_info in predictions.items():
                            if pred_info.get('value') is not None:
                                category, _, _ = predictor.get_aqi_category(pred_info['value'])
                                results_data.append({
                                    'Time Horizon': pred_info['horizon'].upper(),
                                    'AQI Value': round(pred_info['value'], 2),
                                    'Category': category,
                                    'Confidence': pred_info['confidence'],
                                    'Target Time': pred_info['timestamp']
                                })
                        
                        if results_data:
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                    
                    # Export results
                    if st.button("💾 Export Results"):
                        export_data = {
                            'input_data': current_readings,
                            'predictions': predictions,
                            'generated_at': datetime.now().isoformat()
                        }
                        
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"aqi_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        st.info("👈 Please provide sensor readings using the sidebar")
    
    # Additional information
    with st.expander("ℹ️ About AQI Categories"):
        st.markdown("""
        **Air Quality Index Categories:**
        - 🟢 **Good (0-1)**: Air quality is satisfactory
        - 🟡 **Moderate (1-2)**: Air quality is acceptable  
        - 🟠 **Unhealthy for Sensitive (2-3)**: Sensitive groups may experience minor issues
        - 🔴 **Unhealthy (3-4)**: Everyone may experience health effects
        - 🟣 **Very Unhealthy (4-5)**: Health alert: serious health effects
        - 🔴 **Hazardous (5+)**: Emergency conditions: everyone affected
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit • AQI Prediction System")

if __name__ == "__main__":
    main()
