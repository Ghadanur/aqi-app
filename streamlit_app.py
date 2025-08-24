# Loading Models from Separate Repository in Streamlit

## Problem
- **Streamlit app repo**: Contains the UI code
- **Models repo**: Contains `data/oversampling_PubChem_RandomForestClassifier.pkl`
- **Challenge**: `pickle.load(open('data/model.pkl', 'rb'))` won't work because the file doesn't exist in the Streamlit repo

## Solution 1: Download Models at Runtime (Recommended)

Replace your model loading code with dynamic downloading:

```python
import streamlit as st
import pickle
import requests
import os
from pathlib import Path

# Configuration
MODELS_REPO = "YOUR_USERNAME/your-models-repo"  # Replace with actual repo
MODEL_FILE = "oversampling_PubChem_RandomForestClassifier.pkl"

@st.cache_resource
def load_model_from_github():
    """Download and load model from GitHub repository"""
    
    # Create local models directory
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / MODEL_FILE
    
    # Download if not exists
    if not model_path.exists():
        with st.spinner(f"Downloading model: {MODEL_FILE}..."):
            try:
                # Method 1: From GitHub raw content
                raw_url = f"https://raw.githubusercontent.com/{MODELS_REPO}/main/data/{MODEL_FILE}"
                
                response = requests.get(raw_url, timeout=60)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                
                st.success(f"✅ Model downloaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to download model: {str(e)}")
                
                # Alternative: Try from releases
                try:
                    st.info("Trying to download from latest release...")
                    release_url = f"https://api.github.com/repos/{MODELS_REPO}/releases/latest"
                    release_response = requests.get(release_url)
                    
                    if release_response.status_code == 200:
                        release_data = release_response.json()
                        
                        # Find the model file in assets
                        for asset in release_data.get('assets', []):
                            if MODEL_FILE in asset['name']:
                                asset_response = requests.get(asset['browser_download_url'])
                                
                                with open(model_path, 'wb') as f:
                                    f.write(asset_response.content)
                                
                                st.success("✅ Model downloaded from release!")
                                break
                        else:
                            st.error("Model file not found in latest release")
                            return None
                    else:
                        st.error("Could not access releases")
                        return None
                        
                except Exception as e2:
                    st.error(f"❌ All download methods failed: {str(e2)}")
                    return None
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        st.info(f"🎯 Model loaded: {MODEL_FILE}")
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# Usage in your app
def main():
    st.title("My ML App")
    
    # Load model
    model = load_model_from_github()
    
    if model is None:
        st.error("Cannot proceed without model")
        st.stop()
    
    # Your existing prediction code
    st.write("Model loaded successfully!")
    
    # Example prediction
    if st.button("Make Prediction"):
        # Replace with your actual input preparation
        input_X = prepare_input_data()  # Your input preparation function
        
        try:
            predictions = model.predict(input_X)
            st.success(f"Prediction: {predictions}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
```

## Solution 2: Environment-Based Loading

Handle different environments (local vs cloud):

```python
import streamlit as st
import pickle
import requests
import os
from pathlib import Path

@st.cache_resource
def load_model():
    """Load model from different sources based on environment"""
    
    # Local development - model exists in repo
    local_path = Path("data/oversampling_PubChem_RandomForestClassifier.pkl")
    
    if local_path.exists():
        st.info("🏠 Loading model from local file")
        with open(local_path, 'rb') as f:
            return pickle.load(f)
    
    # Production - download from GitHub
    else:
        st.info("☁️ Loading model from GitHub")
        return download_model_from_github()

def download_model_from_github():
    """Download model from GitHub repository"""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    model_file = "oversampling_PubChem_RandomForestClassifier.pkl"
    model_path = models_dir / model_file
    
    if not model_path.exists():
        with st.spinner("Downloading model..."):
            # Your download logic here
            url = f"https://raw.githubusercontent.com/YOUR_USERNAME/models-repo/main/data/{model_file}"
            
            response = requests.get(url)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                f.write(response.content)
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)
```

## Solution 3: Git Submodules (For Complex Cases)

If you have many models and complex dependencies:

```bash
# In your Streamlit repo
git submodule add https://github.com/YOUR_USERNAME/models-repo.git data

# This creates a `data/` folder with your models
# Now your original code works:
# model = pickle.load(open('data/oversampling_PubChem_RandomForestClassifier.pkl', 'rb'))
```

**But remember**: Streamlit Cloud needs to clone with submodules, which requires additional setup.

## Solution 4: Multiple Models Support

If you have multiple models:

```python
@st.cache_resource
def load_multiple_models():
    """Load multiple models"""
    models = {}
    
    model_files = {
        'random_forest': 'oversampling_PubChem_RandomForestClassifier.pkl',
        'xgboost': 'oversampling_PubChem_XGBoostClassifier.pkl',
        # Add more models as needed
    }
    
    for model_name, filename in model_files.items():
        models[model_name] = download_and_load_model(filename)
    
    return models

def download_and_load_model(filename):
    """Download and load a specific model"""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / filename
    
    if not model_path.exists():
        url = f"https://raw.githubusercontent.com/YOUR_USERNAME/models-repo/main/data/{filename}"
        
        with st.spinner(f"Downloading {filename}..."):
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                f.write(response.content)
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Usage
models = load_multiple_models()
selected_model = st.selectbox("Choose Model", list(models.keys()))
prediction = models[selected_model].predict(input_X)
```

## Recommended Approach

**For your case, use Solution 1** because:
- ✅ Works with Streamlit Cloud out of the box
- ✅ No complex git configuration needed
- ✅ Caches models locally after first download
- ✅ Handles download failures gracefully
- ✅ Easy to maintain and update

## Quick Implementation

Replace your existing model loading code:

```python
# OLD CODE:
# model = pickle.load(open('data/oversampling_PubChem_RandomForestClassifier.pkl', 'rb'))

# NEW CODE:
model = load_model_from_github()
if model is None:
    st.error("Failed to load model")
    st.stop()

# Rest of your prediction code remains the same
predictions = model.predict(input_X)
```

## Important Notes

1. **Update your `requirements.txt`** to include `requests`
2. **Replace `YOUR_USERNAME/your-models-repo`** with your actual repository details
3. **Test locally first** before deploying to Streamlit Cloud
4. **Consider model file size** - GitHub has file size limits (100MB for regular files, 2GB with Git LFS)
