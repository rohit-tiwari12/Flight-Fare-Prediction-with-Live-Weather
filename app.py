from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("flight_rf.pkl", "rb"))


# ============================
# PREPROCESSING FUNCTION
# ============================
def preprocess_input(airline, source, destination, date, dep_time, arr_time, stops, duration):

    # Convert date
    journey_date = pd.to_datetime(date, dayfirst=True)
    journey_day = journey_date.day
    journey_month = journey_date.month

    # Dep time
    dep_dt = pd.to_datetime(dep_time)
    dep_hour = dep_dt.hour
    dep_min = dep_dt.minute

    # Arrival time
    arr_dt = pd.to_datetime(arr_time)
    arrival_hour = arr_dt.hour
    arrival_min = arr_dt.minute

    # Duration conversion
    duration = duration.strip().lower()
    dur_hours = 0
    dur_mins = 0

    if "h" in duration:
        dur_hours = int(duration.split("h")[0])
    if "m" in duration:
        if "h" in duration:
            dur_mins = int(duration.split("h")[1].strip().replace("m", ""))
        else:
            dur_mins = int(duration.replace("m", ""))

    # Total stops
    stops_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
    total_stops = stops_map.get(stops, 0)

    # Encoding airline, source, destination exactly like training
    airline_list = ['Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
                    'Jet Airways Business', 'Multiple carriers',
                    'Multiple carriers Premium economy',
                    'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy']

    if airline not in airline_list:
        airline_enc = 0
    else:
        airline_enc = airline_list.index(airline)

    source_list = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
    if source not in source_list:
        source_enc = 0
    else:
        source_enc = source_list.index(source)

    dest_list = ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
    if destination not in dest_list:
        destination_enc = 0
    else:
        destination_enc = dest_list.index(destination)

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


# ============================
# HOME ROUTE
# ============================
@app.route("/")
def home():
    return render_template("index.html")


# ============================
# PREDICT ROUTE
# ============================
@app.route("/predict", methods=["POST"])
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
        airline, source, destination, date, dep_time, arr_time, stops, duration
    )

    prediction = model.predict(final_input)[0]

    return render_template("index.html", prediction_text=f"Predicted Fare: ₹{int(prediction)}")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    app.run(debug=True)
