"""
Step 2: Fetch Historical Weather Data from Open-Meteo API
==========================================================
This script pulls REAL hourly weather data for 6 UP/NCR cities.
Open-Meteo uses ERA5 reanalysis data (the gold standard for historical weather).

No API key needed. Completely free.
"""

import requests   # To send web requests to Open-Meteo
import pandas as pd  # To organize data into tables
import time  # To add small delays between requests (be polite to the server)

# ── 1. Define our cities and their coordinates ──────────────────────────
# These are the 6 cities we're covering in UP/NCR.
# Latitude and Longitude tell Open-Meteo exactly where to get weather from.

CITIES = {
    "Lucknow":   {"lat": 26.85, "lon": 80.95},
    "Noida":     {"lat": 28.53, "lon": 77.39},
    "Ghaziabad": {"lat": 28.67, "lon": 77.43},
    "Agra":      {"lat": 27.18, "lon": 78.01},
    "Firozabad": {"lat": 27.15, "lon": 78.39},
    "Meerut":    {"lat": 28.98, "lon": 77.71},
}

# ── 2. Define the weather variables we want ─────────────────────────────
# These match the features listed in Section 4.3 of the project overview.
# Paper 3 (Jeong et al.) proved these common variables are sufficient.

WEATHER_VARIABLES = [
    "temperature_2m",         # Temperature at 2 meters above ground (°C)
    "relative_humidity_2m",   # Humidity (%)
    "precipitation",          # Rain/snow (mm per hour)
    "wind_speed_10m",         # Wind speed at 10m height (km/h)
    "cloud_cover",            # Cloud coverage (%)
    "surface_pressure",       # Atmospheric pressure (hPa)
]

# ── 3. Define the date range ────────────────────────────────────────────
# We're pulling 2 years of data: 2023 and 2024.
# This gives us ~17,520 hourly data points PER city.

START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# ── 4. The main function that fetches weather for one city ──────────────

def fetch_city_weather(city_name, lat, lon):
    """
    Sends a request to Open-Meteo and returns a pandas DataFrame
    with hourly weather data for the given city.
    """
    # Build the URL — this is like typing a web address with specific instructions
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(WEATHER_VARIABLES),  # Comma-separated list of variables
        "timezone": "Asia/Kolkata",              # IST timezone
    }

    print(f"  Fetching data for {city_name}...")
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"  ❌ Error fetching {city_name}: HTTP {response.status_code}")
        return None

    data = response.json()  # Convert the response into a Python dictionary

    # Extract the hourly data into a pandas DataFrame (a table)
    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "temperature_2m": hourly["temperature_2m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "cloud_cover": hourly["cloud_cover"],
        "surface_pressure": hourly["surface_pressure"],
    })

    # Add the city name as a column so we know which city each row belongs to
    df["city"] = city_name

    print(f"  ✅ {city_name}: {len(df)} hourly records fetched")
    return df

# ── 5. Loop through all cities and combine the data ─────────────────────

def main():
    print("=" * 60)
    print("Fetching historical weather data from Open-Meteo")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Cities: {', '.join(CITIES.keys())}")
    print("=" * 60)

    all_data = []  # This list will hold DataFrames from each city

    for city_name, coords in CITIES.items():
        df = fetch_city_weather(city_name, coords["lat"], coords["lon"])
        if df is not None:
            all_data.append(df)
        time.sleep(1)  # Wait 1 second between requests (polite to the server)

    # Combine all city DataFrames into one big table
    combined = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    output_file = "weather_data.csv"
    combined.to_csv(output_file, index=False)

    print("=" * 60)
    print(f"✅ Done! Saved {len(combined)} total rows to '{output_file}'")
    print(f"   That's {len(combined)} hours of weather data across {len(CITIES)} cities")
    print(f"\nFirst 5 rows preview:")
    print(combined.head().to_string(index=False))
    print("=" * 60)

# ── 6. Run the script ──────────────────────────────────────────────────

if __name__ == "__main__":
    main()
