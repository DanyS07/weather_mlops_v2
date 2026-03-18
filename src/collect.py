import requests, pandas as pd, yaml, os
from datetime import datetime, timedelta

params = yaml.safe_load(open("params.yaml"))

def fetch(lat, lon, name):
    end = datetime.today()
    start = end - timedelta(days=params["data"]["days"])

    url = "https://archive-api.open-meteo.com/v1/archive"

    res = requests.get(url, params={
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
    }).json()

    df = pd.DataFrame({
        "time": res["hourly"]["time"],
        "temp": res["hourly"]["temperature_2m"],
        "humidity": res["hourly"]["relative_humidity_2m"],
        "precip": res["hourly"]["precipitation"],
        "wind": res["hourly"]["wind_speed_10m"]
    })

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{name}.csv", index=False)

for name, coord in params["locations"].items():
    fetch(coord[0], coord[1], name)