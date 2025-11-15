import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import pickle

df = pd.read_excel("Data_Train.xlsx")

df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True, errors='coerce')
df["journey_day"] = df["Date_of_Journey"].dt.day
df["journey_month"] = df["Date_of_Journey"].dt.month

df["Dep_Time"] = pd.to_datetime(df["Dep_Time"], errors='coerce')
df["dep_hour"] = df["Dep_Time"].dt.hour
df["dep_min"] = df["Dep_Time"].dt.minute

df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"], errors='coerce')
df["arrival_hour"] = df["Arrival_Time"].dt.hour
df["arrival_min"] = df["Arrival_Time"].dt.minute

def convert_duration(x):
    x = x.strip()
    hours = 0
    mins = 0
    if "h" in x:
        hours = int(x.split('h')[0])
    if "m" in x:
        if "h" in x:
            mins = int(x.split('h')[1].strip().replace("m", ""))
        else:
            mins = int(x.replace("m", ""))
    return hours, mins

df["duration_hours"] = df["Duration"].apply(lambda x: convert_duration(x)[0])
df["duration_mins"] = df["Duration"].apply(lambda x: convert_duration(x)[1])

df["Total_Stops"] = df["Total_Stops"].replace({
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
})

df.drop(["Route", "Additional_Info", "Date_of_Journey",
         "Dep_Time", "Arrival_Time", "Duration"], axis=1, inplace=True)

encoder = LabelEncoder()
for col in ["Airline", "Source", "Destination"]:
    df[col] = encoder.fit_transform(df[col].astype(str))

df.dropna(inplace=True)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

pickle.dump(model, open("flight_rf.pkl", "wb"))

print("Model retrained using sklearn 1.6.1 and saved correctly!")
