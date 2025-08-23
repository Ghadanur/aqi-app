# quick_start.py - Quick setup script for AQI Streamlit App

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'joblib', 'pycaret', 'lightgbm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        install = input("Install missing packages? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        else:
            logger.error("Cannot proceed without required packages")
            return False
    
    logger.info("✅ All dependencies available")
    return True

def setup_project_structure():
    """Create necessary directories"""
    dirs = ['src/models', 'src/features', 'models', '.streamlit']
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_name}")
    
    # Create __init__.py files for proper imports
    init_files = ['src/__init__.py', 'src/models/__init__.py', 'src/features/__init__.py']
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Package initialization\n")
            logger.info(f"📝 Created: {init_file}")

def check_models():
    """Check if trained models exist"""
    model_files = [
        'models/aqi_current_model.pkl',
        'models/aqi_24h_model.pkl', 
        'models/aqi_48h_model.pkl',
        'models/aqi_72h_model.pkl'
    ]
    
    existing_models = [f for f in model_files if os.path.exists(f)]
    
    if existing_models:
        logger.info(f"✅ Found {len(existing_models)} trained models")
        return True
    else:
        logger.warning("⚠️ No trained models found")
        logger.info("You'll need to train models first before using predictions")
        return False

def create_sample_data():
    """Create sample data file for testing"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample data
    dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    
    sample_data = {
        'timestamp': dates,
        'aqi': np.random.uniform(2.0, 4.0, 24),
        'pm2_5': np.random.uniform(20, 40, 24),
        'pm10': np.random.uniform(35, 55, 24),
        'co': np.random.uniform(0.5, 1.5, 24),
        'so2': np.random.uniform(3, 8, 24),
        'no2': np.random.uniform(15, 30, 24),
        'o3': np.random.uniform(70, 100, 24),
        'temperature': np.random.uniform(25, 35, 24),
        'humidity': np.random.uniform(50, 80, 24),
        'wind_speed': np.random.uniform(3, 10, 24),
        'pressure': np.random.uniform(1000, 1030, 24),
        'uv_index': np.random.uniform(3, 10, 24)
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_aqi_data.csv', index=False)
    logger.info("📊 Created sample_aqi_data.csv for testing")

def create_streamlit_config():
    """Create Streamlit configuration"""
    config_content = """[server]
port = 8501
address = "localhost"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
"""
    
    with open('.streamlit/config.toml', 'w') as f:
        f.write(config_content)
    logger.info("⚙️ Created Streamlit configuration")

def main():
    """Main setup function"""
    print("🚀 AQI Streamlit App Quick Start Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup project structure
    setup_project_structure()
    
    # Check for models
    has_models = check_models()
    
    # Create sample data
    create_sample_data()
    
    # Create config
    create_streamlit_config()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed!")
    print("\n📋 Next Steps:")
    
    if not has_models:
        print("1. Train your models first:")
        print("   python -m src.models.train")
    
    print("2. Run the Streamlit app:")
    print("   streamlit run streamlit_aqi_app.py")
    
    print("\n💡 Tips:")
    print("- Use 'Random Sample' to test the app without real data")
    print("- Upload 'sample_aqi_data.csv' to test CSV functionality") 
    print("- Check the setup guide for troubleshooting")
    
    # Ask if user wants to run the app
    if has_models:
        run_app = input("\n🚀 Run the Streamlit app now? (y/n): ")
        if run_app.lower() == 'y':
            try:
                subprocess.run(["streamlit", "run", "streamlit_aqi_app.py"])
            except KeyboardInterrupt:
                print("\n👋 App stopped by user")
            except File
