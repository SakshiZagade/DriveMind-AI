import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# load dataset
data = pd.read_csv("data/automotive_telemetry_150k.csv")

# features used for driving behaviour detection
features = [
    "speed_kmh",
    "acceleration",
    "brake_pressure",
    "steering_angle",
    "rpm",
    "engine_temp"
]

X = data[features]

# convert label to numeric
y = data["anomaly_label"].map({
    "normal": 0,
    "aggressive_driving": 1
})

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# evaluate model
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

# save model
joblib.dump(model, "models/driving_behavior_model.pkl")

print("Model saved successfully")