import joblib
import pandas as pd

# load trained model
model = joblib.load("models/driving_behavior_model.pkl")

features = [
    "speed_kmh",
    "acceleration",
    "brake_pressure",
    "steering_angle",
    "rpm",
    "engine_temp"
]

def analyze_behavior(df):

    if df.empty:
        return "No data found for this query."

    X = df[features]

    predictions = model.predict(X)

    df["prediction"] = predictions

    aggressive_count = (df["prediction"] == 1).sum()
    normal_count = (df["prediction"] == 0).sum()

    vehicles = df["vehicle_id"].nunique()

    insight = f"""
AI Driving Behaviour Analysis

Vehicles analysed: {vehicles}
Aggressive driving events detected: {aggressive_count}
Normal driving events: {normal_count}

Interpretation:
Frequent high acceleration and brake pressure patterns indicate aggressive driving behaviour.
"""

    return insight