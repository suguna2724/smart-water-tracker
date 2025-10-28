import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('cleaned_global_water_consumption.csv')

# Feature and target selection
X = df[['Per Capita Water Use (Liters per Day)',
        'Agricultural Water Use (%)',
        'Industrial Water Use (%)',
        'Household Water Use (%)',
        'Rainfall Impact (Annual Precipitation in mm)',
        'Groundwater Depletion Rate (%)',
        'Year']]

y = df['Total Water Consumption (Billion Cubic Meters)']

# -----------------------------
# Split, scale, train
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# -----------------------------
# Save model and scaler
# -----------------------------
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully!")
