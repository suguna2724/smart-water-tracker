from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

# ============================================================
# 🌊 SMART WATER TRACKER — FLASK APP (SUSTAINABILITY PROJECT)
# ============================================================

app = Flask(__name__)

# ------------------------------
# 1️⃣ Load Model, Scaler, and Data
# ------------------------------
try:
    model = joblib.load('rf_model.pkl')      # Random Forest model
    scaler = joblib.load('scaler.pkl')       # Scaler for normalization
    data = pd.read_csv('cleaned_global_water_consumption.csv')
    print("✅ Model, Scaler, and Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or data: {e}")

# ------------------------------
# 2️⃣ Homepage Route
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------
# 3️⃣ About Page
# ------------------------------
@app.route('/about')
def about():
    return render_template('about.html')

# ------------------------------
# 4️⃣ Prediction Page (Form + JS API)
# ------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # --- Handle HTML form submission ---
            if request.form:
                country = request.form['country']
                year = int(request.form['year'])

            # --- Handle JS JSON fetch request ---
            elif request.is_json:
                req = request.get_json()
                country = req.get('country')
                year = int(req.get('year'))

            # --- Filter dataset ---
            df_country = data[data['Country'].str.lower() == country.lower()]
            if df_country.empty:
                error_msg = f"❌ Country '{country}' not found in dataset."
                return render_template('predict.html', error=error_msg)

            # --- Use last row for prediction ---
            last_row = df_country.iloc[-1]

            features = np.array([[
                last_row['Per Capita Water Use (Liters per Day)'],
                last_row['Agricultural Water Use (%)'],
                last_row['Industrial Water Use (%)'],
                last_row['Household Water Use (%)'],
                last_row['Rainfall Impact (Annual Precipitation in mm)'],
                last_row['Groundwater Depletion Rate (%)'],
                year
            ]])

            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]

            # --- Generate insight ---
            if prediction > 600:
                insight = f"🌊 {country} is expected to face **high water demand**. Prioritize water-efficient technologies and smart irrigation."
            elif 400 <= prediction <= 600:
                insight = f"💧 {country}'s water usage is **moderate**. Encourage sustainable farming and recycling initiatives."
            else:
                insight = f"🌿 {country} shows **efficient water use**. Continue conservation awareness programs and resource planning."

            # --- Render or return JSON ---
            if request.is_json:
                return jsonify({'prediction': float(prediction), 'insight': insight})
            else:
                return render_template('predict.html', prediction=prediction, insight=insight)

        except Exception as e:
            return render_template('predict.html', error=f"⚠️ Error: {str(e)}")

    # If GET request → show prediction page
    return render_template('predict.html')

# ------------------------------
# 5️⃣ Custom Error Handlers
# ------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

# ------------------------------
# 6️⃣ Run Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
