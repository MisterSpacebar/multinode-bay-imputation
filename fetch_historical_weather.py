"""Fetch 2021-2024 Miami daily weather from Open-Meteo (ERA5) and save."""
import urllib.request
import json
import pandas as pd
from pathlib import Path

URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=25.796&longitude=-80.287"
    "&start_date=2021-01-01&end_date=2024-12-31"
    "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
    "&precipitation_unit=inch"
    "&timezone=America%2FNew_York"
)

print("Fetching Miami 2021-2024 weather from Open-Meteo (ERA5)...")
with urllib.request.urlopen(URL, timeout=30) as r:
    data = json.loads(r.read())

daily = data["daily"]
df = pd.DataFrame({
    "date":     pd.to_datetime(daily["time"]),
    "temp_min": daily["temperature_2m_min"],
    "temp_max": daily["temperature_2m_max"],
    "rain_in":  daily["precipitation_sum"],
})
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"]   = df["date"].dt.day
df = df[["year", "month", "day", "temp_min", "temp_max", "rain_in"]]
df["temp_min"] = df["temp_min"].round(2)
df["temp_max"] = df["temp_max"].round(2)
df["rain_in"]  = df["rain_in"].round(3)

out = Path("water_data/rainfall/miami_airport_daily_2021_2024.csv")
df.to_csv(out, index=False)

first = df.iloc[0]
last  = df.iloc[-1]
print(f"Saved {len(df)} rows to {out}")
print(df.head(5).to_string(index=False))
print()
print(f"Date range:  {int(first.year)}-{int(first.month):02d}-{int(first.day):02d}"
      f"  to  {int(last.year)}-{int(last.month):02d}-{int(last.day):02d}")
print(f"Rain range:  {df.rain_in.min():.3f} - {df.rain_in.max():.3f} in/day")
print(f"Temp range:  {df.temp_min.min():.1f} - {df.temp_max.max():.1f} C")
print(f"Missing:     {df.isnull().sum().sum()} values")
