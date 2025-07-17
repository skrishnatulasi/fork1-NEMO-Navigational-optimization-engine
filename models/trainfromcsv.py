import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import numpy as np

# Load fallback CSV
df = pd.read_csv(os.path.join('data', 'sst_fallback.csv'))

# Simulate a 'catch' column (for demo, make it a function of sst, wind_speed, salinity)
df['catch'] = (
    10
    + 2 * (df['sst'] - 27)
    - 1.5 * abs(df['salinity'] - 35)
    - 0.5 * df['wind_speed']
    + np.random.normal(0, 1, len(df))
)
df['catch'] = df['catch'].clip(lower=0)

# Features and target
X = df[['latitude', 'longitude', 'sst', 'salinity', 'wind_speed']]
y = df['catch']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/predictor.pkl')
print("Model trained and saved to models/predictor.pkl")
