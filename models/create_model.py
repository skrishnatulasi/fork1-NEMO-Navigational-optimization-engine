import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Create sample training data for Tamil Nadu fishing
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Tamil Nadu coastal coordinates
    latitudes = np.random.uniform(8.0, 13.5, n_samples)
    longitudes = np.random.uniform(77.0, 80.5, n_samples)
    
    # Environmental features
    sst = np.random.uniform(24, 30, n_samples)  # Sea surface temperature
    salinity = np.random.uniform(32, 38, n_samples)
    wind_speed = np.random.uniform(2, 15, n_samples)
    depth = np.random.uniform(10, 200, n_samples)
    
    # Create realistic catch predictions based on environmental factors
    catch_yield = (
        (sst - 27) ** 2 * -0.5 +  # Optimal around 27°C
        (salinity - 35) ** 2 * -0.3 +  # Optimal around 35 ppt
        (wind_speed - 8) ** 2 * -0.2 +  # Optimal around 8 m/s
        np.random.normal(0, 2, n_samples) +  # Random noise
        15  # Base catch
    )
    
    # Ensure positive values
    catch_yield = np.maximum(catch_yield, 1)
    
    data = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'sst': sst,
        'salinity': salinity,
        'wind_speed': wind_speed,
        'depth': depth,
        'catch_yield': catch_yield
    })
    
    return data

def train_model():
    print("🤖 Creating sample fishing data...")
    data = create_sample_data()
    
    print("📊 Training ML model...")
    features = ['latitude', 'longitude', 'sst', 'salinity', 'wind_speed', 'depth']
    X = data[features]
    y = data['catch_yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"✅ Model trained! R² Score: {score:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/predictor.pkl')
    print("💾 Model saved to models/predictor.pkl")
    
    return model

if __name__ == "__main__":
    train_model()
