✈️ Flight Fare Prediction with Live Weather | Machine Learning + Flask

This project predicts flight ticket prices based on various travel features such as
airline, journey date, number of stops, duration, source & destination city — enhanced with live real-time weather data using the OpenWeather API.

It is built using:

Python

Machine Learning (DecisionTreeRegressor)

Flask Web App

HTML + CSS (Custom UI)

OpenWeather API Integration


🚀 Features
✔ Prediction Based on:

Airline

Source & Destination

Number of stops

Journey date

Departure & arrival time

Duration

Live weather conditions at the source airport:

Temperature

Visibility

Wind speed

Rain (Yes/No)

✔ Simple Flask Web Interface
✔ Clean UI with custom CSS
✔ Fully working ML pipeline
✔ Model trained on enriched features (16 total features)



📂 Project Structure

Flight_prediction/
│── app.py
│── train_model_weather.py
│── flight_rf.pkl
│── requirements.txt
│── Data_Train.xlsx
│── Test_set.xlsx
│── templates/
│     └── index.html
│── static/
│     └── css/
│          └── flight.css

1️⃣ Clone the Repository
git clone https://github.com/rohit-tiwari12/Flight-Fare-Prediction.git
cd Flight-Fare-Prediction

2️⃣ Install Dependencies

pip install -r requirements.txt





🌦 Weather Feature

The app fetches live weather from the source city using OpenWeather API.

It extracts:

Temperature (°C)

Visibility (meters)

Wind Speed (m/s)

Rain (Yes / No)

These values influence flight fare prediction.

📊 Output Example

After submitting details, the app displays:

Predicted Fare

Live Weather Conditions

Temperature

Visibility

Wind Speed

Rain Status





🛠 Future Enhancements

🔹 Multi-model support (Random Forest, XGBoost)
🔹 Add airport codes
🔹 Add fare trends chart
🔹 Travel season impact analysis
🔹 Deploy on Render / Railway

🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss what you would like to change.


