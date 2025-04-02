import os
import json
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

# 🔥 Load Firebase Credentials
cred = credentials.Certificate("firebase-key.json")  # Ensure this file exists
firebase_admin.initialize_app(cred)
db = firestore.client()

# 📍 Load Hospital Data from CSV
csv_path = "hospitals_bloodbanks_real.csv"
df = pd.read_csv(csv_path)

# ✅ Ensure necessary columns exist
required_columns = {"Name", "Latitude", "Longitude", "Blood Type", "Availability (Units)", "Contact"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file is missing required columns: {required_columns - set(df.columns)}")

# 🔮 Load Blood Demand History for Prophet Forecasting
history_csv_path = "blood_demand_history.csv"
df_history = pd.read_csv(history_csv_path)
df_history["Date"] = pd.to_datetime(df_history["Date"], format="%d-%m-%Y", errors="coerce")

app = Flask(__name__)
CORS(app)  # Enable CORS for Android app access

FCM_URL = "https://fcm.googleapis.com/v1/projects/bb11-cfc8e/messages:send"  # Replace with your Firebase project ID

# 🔮 **Blood Demand Forecasting with Prophet**
@app.route('/forecast_blood_demand', methods=['POST'])
def forecast_blood_demand():
    try:
        data = request.get_json()
        hospital = data.get("hospital")
        blood_type = data.get("blood_type")
        months = data.get("months", 3)

        if not hospital or not blood_type:
            return jsonify({"error": "Missing parameters"}), 400

        df_filtered = df_history[(df_history["Hospital"] == hospital) & (df_history["Blood Type"] == blood_type)]
        if df_filtered.empty or len(df_filtered) < 5:
            return jsonify({"error": "Not enough historical data"}), 400

        df_prophet = df_filtered.rename(columns={"Date": "ds", "Demand": "y"})
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=months * 30, freq='D')
        forecast = model.predict(future)

        forecast["month"] = forecast["ds"].dt.to_period("M")
        monthly_forecast = forecast.groupby("month")["yhat"].sum().reset_index()

        response = {
            "hospital": hospital,
            "blood_type": blood_type,
            "forecast": {str(row["month"]): int(row["yhat"]) for _, row in monthly_forecast.iterrows()},
            "status": "success"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# ✅ **Find Nearest Hospitals using KNN**
@app.route('/nearest_hospitals', methods=['POST'])
def get_nearest_hospitals():
    data = request.get_json()
    user_lat, user_long, blood_type = data.get('latitude'), data.get('longitude'), data.get('blood_type')

    if None in (user_lat, user_long, blood_type):
        return jsonify({"error": "Missing parameters"}), 400

    df_filtered = df[df["Blood Type"].str.strip().str.upper() == blood_type.strip().upper()].copy()
    if df_filtered.empty:
        return jsonify({"error": "No hospitals found with the requested blood type."})

    user_location = np.radians([[user_lat, user_long]])
    hospital_coords = np.radians(df_filtered[["Latitude", "Longitude"]].values)
    nn = NearestNeighbors(n_neighbors=5, metric="haversine")
    nn.fit(hospital_coords)
    distances, indices = nn.kneighbors(user_location)
    distances_km = distances[0] * 6371.0

    nearest_hospitals = df_filtered.iloc[indices[0]].copy()
    nearest_hospitals["Distance"] = distances_km

    return jsonify(nearest_hospitals[["Name", "Blood Type", "Availability (Units)", "Distance", "Contact"]].to_dict(orient="records"))

# 🔥 **Find Best Donation Hospitals using Decision Tree & Weighted Scoring**
@app.route('/best_donation_hospital', methods=['POST'])
def get_best_donation_hospital():
    data = request.get_json()
    user_lat, user_long, blood_type = data.get('latitude'), data.get('longitude'), data.get('blood_type')

    if None in (user_lat, user_long, blood_type):
        return jsonify({"error": "Missing parameters"}), 400

    df_filtered = df[df["Blood Type"].str.strip().str.upper() == blood_type.strip().upper()].copy()
    if df_filtered.empty:
        return jsonify({"error": "No hospitals found with the requested blood type."})

    user_location = np.radians([[user_lat, user_long]])
    hospital_coords = np.radians(df_filtered[["Latitude", "Longitude"]].values)
    nn = NearestNeighbors(n_neighbors=10, metric="haversine")
    nn.fit(hospital_coords)
    distances, indices = nn.kneighbors(user_location)
    distances_km = distances[0] * 6371.0

    nearest_hospitals = df_filtered.iloc[indices[0]].copy()
    nearest_hospitals["Distance"] = distances_km

    # 🔥 Apply Weighted Scoring Model
    nearest_hospitals["Total Score"] = 100 - (nearest_hospitals["Availability (Units)"] * 2) + (1 / nearest_hospitals["Distance"]) * 100

    best_hospitals = nearest_hospitals.nlargest(5, "Total Score")

    return jsonify(best_hospitals[["Name", "Blood Type", "Availability (Units)", "Distance", "Contact"]].to_dict(orient="records"))


# ✅ **Fetch Hospitals from Demand Data**
@app.route('/get_hospitals_from_demand', methods=['GET'])
def get_hospitals_from_demand():
    unique_hospitals = df_history["Hospital"].dropna().unique().tolist()
    if not unique_hospitals:
        return jsonify({"error": "No hospital data available"}), 404
    
    return jsonify({"hospitals": unique_hospitals})

@app.route('/request_blood', methods=['POST'])
def request_blood():
    data = request.get_json()
    
    if not all(k in data for k in ("user_name", "phone", "blood_type", "hospital_id", "fcm_token")):
        return jsonify({"error": "Missing parameters"}), 400

    request_ref = db.collection("BloodRequests").document()
    request_ref.set({
        "user_name": data["user_name"],
        "phone": data["phone"],
        "blood_type": data["blood_type"],
        "hospital_id": data["hospital_id"],
        "status": "Pending",
        "fcm_token": data["fcm_token"],
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return jsonify({"message": "Blood request submitted successfully!"})

# ✅ **API Home Route**
@app.route('/')
def home():
    return jsonify({"message": "Blood Bank API is running! 🚀"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)