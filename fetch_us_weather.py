"""
Part B: Fetch US Weather Data & Merge with EAGLE-I Outage Data
================================================================
1. Downloads county centroid coordinates (US Census Bureau)
2. Defines representative weather cities per state
3. Assigns each county to its nearest representative city
4. Fetches 2023 hourly weather from Open-Meteo for each city
5. Merges weather with outage data
6. Saves the complete training dataset

Input:  data/eagle_i_filtered.csv
Output: data/us_training_data.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import io
import os

# ══════════════════════════════════════════════════════════════════════
# 1. REPRESENTATIVE CITIES — coordinates for weather fetching
# ══════════════════════════════════════════════════════════════════════

REPRESENTATIVE_CITIES = {
    "Texas": [
        {"name": "Houston",     "lat": 29.76, "lon": -95.37},  # humid Gulf coast
        {"name": "Dallas",      "lat": 32.78, "lon": -96.80},  # inland north
        {"name": "San_Antonio", "lat": 29.42, "lon": -98.49},  # central south
        {"name": "El_Paso",     "lat": 31.77, "lon": -106.44}, # desert west
    ],
    "Arizona": [
        {"name": "Phoenix",     "lat": 33.45, "lon": -112.07}, # extreme desert heat
        {"name": "Tucson",      "lat": 32.22, "lon": -110.97}, # monsoon belt
        {"name": "Flagstaff",   "lat": 35.20, "lon": -111.65}, # cooler highlands
    ],
    "Louisiana": [
        {"name": "New_Orleans", "lat": 29.95, "lon": -90.07},  # Gulf coast
        {"name": "Baton_Rouge", "lat": 30.45, "lon": -91.19},  # central
        {"name": "Shreveport",  "lat": 32.53, "lon": -93.75},  # northwest
    ],
    "Mississippi": [
        {"name": "Jackson",     "lat": 32.30, "lon": -90.18},  # central
        {"name": "Gulfport",    "lat": 30.37, "lon": -89.09},  # Gulf coast
        {"name": "Tupelo",      "lat": 34.26, "lon": -88.70},  # northeast
    ],
    "Oklahoma": [
        {"name": "Oklahoma_City", "lat": 35.47, "lon": -97.52}, # central
        {"name": "Tulsa",         "lat": 36.15, "lon": -95.99}, # northeast
        {"name": "Lawton",        "lat": 34.60, "lon": -98.39}, # southwest
    ],
    "Florida": [
        {"name": "Miami",        "lat": 25.76, "lon": -80.19},  # south tropical
        {"name": "Tampa",        "lat": 27.95, "lon": -82.46},  # central Gulf
        {"name": "Jacksonville", "lat": 30.33, "lon": -81.66},  # northeast
        {"name": "Pensacola",    "lat": 30.44, "lon": -87.22},  # panhandle
    ],
}

# State FIPS prefixes (for matching counties to states)
STATE_FIPS_PREFIX = {
    "Texas": 48,
    "Arizona": 4,
    "Louisiana": 22,
    "Mississippi": 28,
    "Oklahoma": 40,
    "Florida": 12,
}

# Weather variables to fetch (same as our UP/NCR data)
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
]

# ══════════════════════════════════════════════════════════════════════
# 2. DOWNLOAD COUNTY CENTROID COORDINATES
# ══════════════════════════════════════════════════════════════════════

def download_county_centroids():
    """
    Download county centroid lat/lon coordinates.
    Tries multiple public sources for reliability.
    Returns a DataFrame with columns: fips_code, county_lat, county_lon
    """

    # Source 1: GitHub dataset (fast, reliable)
    github_url = "https://raw.githubusercontent.com/btskinner/spatial/master/data/county_centers.csv"
    print(f"  Downloading county centroids from GitHub (btskinner/spatial)...")

    try:
        response = requests.get(github_url, timeout=30)
        response.raise_for_status()

        gaz = pd.read_csv(io.StringIO(response.text))

        # This dataset has: fips, clon10, clat10 (2010 census centroids)
        centroids = pd.DataFrame({
            "fips_code": gaz["fips"].astype(int),
            "county_lat": gaz["clat10"].astype(float),
            "county_lon": gaz["clon10"].astype(float),
        })

        print(f"  ✅ Downloaded centroids for {len(centroids)} counties")
        return centroids

    except Exception as e:
        print(f"  ⚠️ GitHub source failed: {e}")

    # Source 2: US Census Bureau gazetteer (fallback)
    census_url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.txt"
    print(f"  Trying Census Bureau as fallback...")

    try:
        response = requests.get(census_url, timeout=60)
        response.raise_for_status()

        gaz = pd.read_csv(io.StringIO(response.text), sep="\t")
        gaz.columns = gaz.columns.str.strip()

        centroids = pd.DataFrame({
            "fips_code": gaz["GEOID"].astype(int),
            "county_lat": gaz["INTPTLAT"].astype(float),
            "county_lon": gaz["INTPTLONG"].astype(float),
        })

        print(f"  ✅ Downloaded centroids for {len(centroids)} counties")
        return centroids

    except Exception as e:
        print(f"  ❌ Census fallback also failed: {e}")
        print(f"  Cannot proceed without county coordinates.")
        return None


# ══════════════════════════════════════════════════════════════════════
# 3. ASSIGN COUNTIES TO NEAREST REPRESENTATIVE CITY
# ══════════════════════════════════════════════════════════════════════

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def assign_counties_to_cities(outage_df, centroids_df):
    """
    For each county in the outage data, find the nearest representative city.
    Returns a mapping: fips_code → (city_name, state)
    """
    unique_counties = outage_df[["fips_code", "state"]].drop_duplicates()
    print(f"\n  Assigning {len(unique_counties)} counties to nearest representative cities...")

    # Merge county centroids
    if centroids_df is not None:
        unique_counties = unique_counties.merge(centroids_df, on="fips_code", how="left")
        missing = unique_counties["county_lat"].isna().sum()
        if missing > 0:
            print(f"  ⚠️ {missing} counties missing centroids — will use state primary city")
    else:
        unique_counties["county_lat"] = np.nan
        unique_counties["county_lon"] = np.nan

    assignments = {}

    for _, row in unique_counties.iterrows():
        fips = row["fips_code"]
        state = row["state"]
        cities = REPRESENTATIVE_CITIES[state]

        if pd.notna(row["county_lat"]):
            # Find nearest city by distance
            best_city = None
            best_dist = float("inf")
            for city in cities:
                dist = haversine_distance(
                    row["county_lat"], row["county_lon"],
                    city["lat"], city["lon"]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_city = city["name"]
            assignments[fips] = best_city
        else:
            # Fallback: use the first (primary) city for the state
            assignments[fips] = cities[0]["name"]

    # Print assignment summary
    for state in REPRESENTATIVE_CITIES:
        state_counties = {k: v for k, v in assignments.items()
                         if unique_counties[unique_counties["fips_code"] == k]["state"].values[0] == state}
        city_counts = pd.Series(state_counties.values()).value_counts()
        print(f"  {state}: {dict(city_counts)}")

    return assignments


# ══════════════════════════════════════════════════════════════════════
# 4. FETCH WEATHER DATA FROM OPEN-METEO
# ══════════════════════════════════════════════════════════════════════

def fetch_city_weather(city_name, lat, lon, state):
    """Fetch 2023 hourly weather for one city from Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "America/Chicago",  # Central US time (covers most of our states)
    }

    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        print(f"  ❌ Failed for {city_name}: HTTP {response.status_code}")
        return None

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "hour_timestamp": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "cloud_cover": hourly["cloud_cover"],
        "surface_pressure": hourly["surface_pressure"],
    })

    df["weather_city"] = city_name
    df["weather_state"] = state

    return df


def fetch_all_weather():
    """Fetch weather for all representative cities."""
    all_weather = []
    total_cities = sum(len(cities) for cities in REPRESENTATIVE_CITIES.values())
    fetched = 0

    print(f"\nFetching weather for {total_cities} representative cities from Open-Meteo...")
    print(f"Date range: 2023-01-01 to 2023-12-31\n")

    for state, cities in REPRESENTATIVE_CITIES.items():
        for city in cities:
            fetched += 1
            print(f"  [{fetched}/{total_cities}] {city['name']}, {state}...", end="", flush=True)

            df = fetch_city_weather(city["name"], city["lat"], city["lon"], state)

            if df is not None:
                all_weather.append(df)
                print(f" ✅ {len(df)} hours")
            else:
                print(f" ❌ FAILED")

            time.sleep(1.5)  # Be polite to the API

    combined = pd.concat(all_weather, ignore_index=True)
    print(f"\n✅ Total weather records: {len(combined):,}")
    return combined


# ══════════════════════════════════════════════════════════════════════
# 5. MERGE OUTAGE DATA WITH WEATHER DATA
# ══════════════════════════════════════════════════════════════════════

def merge_outage_and_weather(outage_df, weather_df, county_to_city):
    """
    Merge outage records with their corresponding weather data.
    Each county is matched to its nearest representative city's weather.
    """
    print("\nMerging outage data with weather data...")

    # Add the assigned weather city to the outage dataframe
    outage_df["weather_city"] = outage_df["fips_code"].map(county_to_city)

    # Parse hour_timestamp in outage data
    outage_df["hour_timestamp"] = pd.to_datetime(outage_df["hour_timestamp"])

    # Merge on weather_city + hour_timestamp
    merged = outage_df.merge(
        weather_df,
        on=["weather_city", "hour_timestamp"],
        how="inner",
        suffixes=("", "_weather")
    )

    # Drop redundant state column from weather
    if "weather_state" in merged.columns:
        merged = merged.drop(columns=["weather_state"])

    print(f"  Outage rows:  {len(outage_df):,}")
    print(f"  Merged rows:  {len(merged):,}")

    unmatched = len(outage_df) - len(merged)
    if unmatched > 0:
        print(f"  ⚠️ Unmatched rows: {unmatched:,} (timestamp misalignment between outage and weather data)")
        print(f"     This can happen due to timezone differences. Keeping matched rows only.")

    return merged


# ══════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Part B: Fetch US Weather & Merge with Outage Data")
    print("=" * 60)

    # Step 1: Load filtered outage data
    print("\n── Step 1: Loading filtered outage data ──")
    outage_df = pd.read_csv("data/eagle_i_filtered.csv")
    print(f"  Loaded {len(outage_df):,} rows, {outage_df['fips_code'].nunique()} counties")

    # Step 2: Download county centroids
    print("\n── Step 2: Getting county coordinates ──")
    centroids = download_county_centroids()

    # Step 3: Assign counties to nearest weather city
    print("\n── Step 3: Assigning counties to weather cities ──")
    county_to_city = assign_counties_to_cities(outage_df, centroids)

    # Step 4: Fetch weather data
    print("\n── Step 4: Fetching weather data ──")
    weather_df = fetch_all_weather()

    # Step 5: Merge
    print("\n── Step 5: Merging outage + weather ──")
    merged = merge_outage_and_weather(outage_df, weather_df, county_to_city)

    # Step 6: Validate and save
    print("\n── Step 6: Validation & Save ──")

    # Check for nulls in critical columns
    null_counts = merged[WEATHER_VARIABLES].isnull().sum()
    if null_counts.any():
        print(f"  ⚠️ Null values in weather columns:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"     {col}: {count} nulls")
        print(f"  Dropping rows with null weather values...")
        before = len(merged)
        merged = merged.dropna(subset=WEATHER_VARIABLES)
        print(f"  Dropped {before - len(merged)} rows")

    # Final statistics
    print(f"\n=== FINAL DATASET STATISTICS ===")
    print(f"  Total rows:      {len(merged):,}")
    print(f"  Counties:        {merged['fips_code'].nunique()}")
    print(f"  States:          {sorted(merged['state'].unique())}")
    print(f"  Date range:      {merged['hour_timestamp'].min()} → {merged['hour_timestamp'].max()}")
    print(f"  Outage rate:     {merged['outage'].mean()*100:.1f}%")
    print(f"  Columns:         {list(merged.columns)}")

    # Per-state check
    print(f"\n  Per-state outage rates:")
    for state in sorted(merged["state"].unique()):
        state_df = merged[merged["state"] == state]
        print(f"    {state:<15}: {len(state_df):>10,} rows | outage rate: {state_df['outage'].mean()*100:.1f}%")

    # Save
    output_file = "data/us_training_data.csv"
    merged.to_csv(output_file, index=False)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"✅ DONE! Saved to {output_file} ({file_size_mb:.1f} MB)")
    print(f"\nSample rows:")
    print(merged.head(3).to_string(index=False))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
