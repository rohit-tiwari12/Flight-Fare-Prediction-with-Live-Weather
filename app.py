from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import pandas as pd
import numpy as np
import requests
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("flight_rf.pkl", "rb"))

# OpenWeather API Key
# For local testing, paste your API key below
# For Render deployment, use environment variable
API_KEY = os.getenv("84317563a4c0f5f8085cd57391237a89", "84317563a4c0f5f8085cd57391237a89")


# ==========================================
# WEATHER FUNCTION
# ==========================================
def get_weather(city):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={API_KEY}&units=metric"
        )

        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()

            return {
                "temperature": round(data["main"]["temp"], 1),
                "humidity": data["main"]["humidity"],
                "condition": data["weather"][0]["main"]
            }

    except Exception as e:
        print("Weather API Error:", e)

    return None


# ==========================================
# PREPROCESSING FUNCTION
# ==========================================
def preprocess_input(
    airline,
    source,
    destination,
    date,
    dep_time,
    arr_time,
    stops,
    duration
):

    # Journey Date
    journey_date = pd.to_datetime(date)

    journey_day = journey_date.day
    journey_month = journey_date.month

    # Departure Time
    dep_dt = pd.to_datetime(dep_time)

    dep_hour = dep_dt.hour
    dep_min = dep_dt.minute

    # Arrival Time
    arr_dt = pd.to_datetime(arr_time)

    arrival_hour = arr_dt.hour
    arrival_min = arr_dt.minute

    # Duration
    duration = duration.strip().lower()

    dur_hours = 0
    dur_mins = 0

    if "h" in duration:
        dur_hours = int(duration.split("h")[0])

    if "m" in duration:
        if "h" in duration:
            mins_part = duration.split("h")[1].strip().replace("m", "")
            dur_mins = int(mins_part) if mins_part else 0
        else:
            dur_mins = int(duration.replace("m", ""))

    # Stops
    total_stops = int(stops)

    # Airline Encoding
    airline_list = [
        'Air Asia',
        'Air India',
        'GoAir',
        'IndiGo',
        'Jet Airways',
        'Jet Airways Business',
        'Multiple carriers',
        'Multiple carriers Premium economy',
        'SpiceJet',
        'Trujet',
        'Vistara',
        'Vistara Premium economy'
    ]

    airline_enc = (
        airline_list.index(airline)
        if airline in airline_list
        else 0
    )

    # Source Encoding
    source_list = [
        'Banglore',
        'Chennai',
        'Delhi',
        'Kolkata',
        'Mumbai'
    ]

    source_enc = (
        source_list.index(source)
        if source in source_list
        else 0
    )

    # Destination Encoding
    dest_list = [
        'Banglore',
        'Cochin',
        'Delhi',
        'Hyderabad',
        'Kolkata',
        'New Delhi'
    ]

    destination_enc = (
        dest_list.index(destination)
        if destination in dest_list
        else 0
    )

    final_features = np.array([[
        airline_enc,
        source_enc,
        destination_enc,
        total_stops,
        journey_day,
        journey_month,
        dep_hour,
        dep_min,
        arrival_hour,
        arrival_min,
        dur_hours,
        dur_mins
    ]])

    return final_features


# ==========================================
# HOME ROUTE
# ==========================================
@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


# ==========================================
# PREDICT ROUTE
# ==========================================
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():

    airline = request.form["airline"]
    source = request.form["source"]
    destination = request.form["destination"]
    date = request.form["date"]
    dep_time = request.form["dep_time"]
    arr_time = request.form["arr_time"]
    stops = request.form["stops"]
    duration = request.form["duration"]

    final_input = preprocess_input(
        airline,
        source,
        destination,
        date,
        dep_time,
        arr_time,
        stops,
        duration
    )

    prediction = model.predict(final_input)[0]

    weather = get_weather(source)

    weather_icons = {
        "Clear": "☀️",
        "Clouds": "☁️",
        "Rain": "🌧️",
        "Thunderstorm": "⛈️",
        "Snow": "❄️",
        "Mist": "🌫️",
        "Haze": "🌫️"
    }

    weather_icon = "🌍"

    if weather:
        weather_icon = weather_icons.get(
            weather["condition"],
            "🌍"
        )

    return render_template(
        "index.html",
        prediction_text=f"Predicted Fare: ₹{int(prediction)}",
        weather=weather,
        weather_icon=weather_icon
    )


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)