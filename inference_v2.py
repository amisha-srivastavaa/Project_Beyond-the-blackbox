"""
Phase 3 v2: Enhanced Inference for UP/NCR
================================================================
Re-fetches Indian weather with expanded variables, engineers all
v2 features, and runs the enhanced model.

Output: data/up_predictions_v2.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import time
import os

# ── Cities ──
CITIES = {
    "Lucknow":   {"lat": 26.85, "lon": 80.95},
    "Noida":     {"lat": 28.57, "lon": 77.32},
    "Ghaziabad": {"lat": 28.67, "lon": 77.42},
    "Agra":      {"lat": 27.18, "lon": 78.02},
    "Firozabad": {"lat": 27.15, "lon": 78.39},
    "Meerut":    {"lat": 28.98, "lon": 77.71},
}

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "cloud_cover", "surface_pressure",
    "wind_gusts_10m", "dewpoint_2m", "shortwave_radiation", "weather_code",
]

FEATURES_V2 = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "cloud_cover", "surface_pressure",
    "wind_gusts_10m", "dewpoint_2m", "shortwave_radiation",
    "hour_of_day", "day_of_week", "month",
    "is_summer", "is_monsoon", "is_peak_hour", "is_thunderstorm",
    "heat_index", "gust_ratio", "dewpoint_depression",
    "temp_x_humidity", "rain_x_wind",
    "rolling_avg_temp_24h", "rolling_max_temp_24h",
    "temp_change_3h", "pressure_change_3h",
    "consecutive_hot_hours",
]


def calculate_heat_index(T_celsius, RH):
    T = (T_celsius * 9 / 5) + 32
    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    full_hi = (
        -42.379 + 2.04901523 * T + 10.14333127 * RH
        - 0.22475541 * T * RH - 0.00683783 * T * T
        - 0.05481717 * RH * RH + 0.00122874 * T * T * RH
        + 0.00085282 * T * RH * RH - 0.00000199 * T * T * RH * RH
    )
    HI_final_F = np.where(HI >= 80, full_hi, HI)
    HI_final_F = np.where(T < 40, T, HI_final_F)
    return (HI_final_F - 32) * 5 / 9


def classify_risk(prob):
    if prob >= 0.70:
        return "CRITICAL"
    elif prob >= 0.50:
        return "HIGH"
    elif prob >= 0.30:
        return "MODERATE"
    else:
        return "LOW"


def main():
    print("=" * 60)
    print("Enhanced Inference: UP/NCR Outage Risk (v2 Model)")
    print("=" * 60)

    # ── 1. Fetch Indian weather with expanded variables ──
    print("\nFetching expanded weather for Indian cities...")
    all_dfs = []
    for city_name, coords in CITIES.items():
        print(f"  {city_name}...", end="", flush=True)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(url, params=params, timeout=60)
        data = resp.json()
        hourly = data["hourly"]

        df = pd.DataFrame({"hour_timestamp": pd.to_datetime(hourly["time"])})
        for var in WEATHER_VARS:
            df[var] = hourly.get(var)
        df["city"] = city_name
        all_dfs.append(df)
        print(f" ✅ {len(df)} hours")
        time.sleep(1.5)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(df):,} rows")

    # ── 2. Engineer features ──
    print("\nEngineering features...")

    df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"])
    df["hour_of_day"] = df["hour_timestamp"].dt.hour
    df["day_of_week"] = df["hour_timestamp"].dt.dayofweek
    df["month"] = df["hour_timestamp"].dt.month

    # Indian seasons
    df["is_summer"] = df["month"].isin([4, 5, 6]).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].isin([6, 7, 8, 9, 10, 18, 19, 20, 21, 22]).astype(int)

    df["heat_index"] = calculate_heat_index(df["temperature_2m"], df["relative_humidity_2m"])

    df["gust_ratio"] = np.where(
        df["wind_speed_10m"] > 0,
        df["wind_gusts_10m"] / df["wind_speed_10m"], 0
    )
    df["dewpoint_depression"] = df["temperature_2m"] - df["dewpoint_2m"]
    df["temp_x_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["rain_x_wind"] = df["precipitation"] * df["wind_speed_10m"]

    storm_codes = [95, 96, 99, 65, 67, 75, 77, 85, 86]
    df["is_thunderstorm"] = df["weather_code"].isin(storm_codes).astype(int)

    # Rolling features per city
    df = df.sort_values(["city", "hour_timestamp"]).reset_index(drop=True)
    print("  Computing rolling averages...")
    for city in df["city"].unique():
        mask = df["city"] == city
        idx = df[mask].index
        df.loc[idx, "rolling_avg_temp_24h"] = df.loc[idx, "temperature_2m"].rolling(24, min_periods=1).mean()
        df.loc[idx, "rolling_max_temp_24h"] = df.loc[idx, "temperature_2m"].rolling(24, min_periods=1).max()
        df.loc[idx, "temp_change_3h"] = df.loc[idx, "temperature_2m"].diff(3)
        df.loc[idx, "pressure_change_3h"] = df.loc[idx, "surface_pressure"].diff(3)
        hot = (df.loc[idx, "temperature_2m"] > 35).astype(int)
        streaks = hot.groupby((hot != hot.shift()).cumsum()).cumsum()
        df.loc[idx, "consecutive_hot_hours"] = streaks * hot

    df = df.dropna(subset=FEATURES_V2)
    print(f"  Rows ready: {len(df):,}")

    # ── 3. Load model and predict ──
    print("\nLoading enhanced model...")
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_model_v2.json")

    X = df[FEATURES_V2]
    probabilities = model.predict_proba(X)[:, 1]

    df["risk_score"] = (probabilities * 100).round(1)
    df["risk_level"] = df["risk_score"].apply(lambda x: classify_risk(x / 100))
    df["predicted_outage"] = (probabilities >= 0.55).astype(int)  # optimized via threshold analysis

    # ── 4. Results ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Enhanced v2 Model)")
    print("=" * 60)

    print(f"\n  Risk Level Distribution:")
    for level in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        count = (df["risk_level"] == level).sum()
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {level:>10}: {count:>8,} hours ({pct:5.1f}%) {bar}")

    print(f"\n  Per-City Average Risk Score:")
    print(f"  {'City':<15} {'Avg Risk':>10} {'Peak Risk':>10} {'CRITICAL hrs':>14} {'HIGH hrs':>10}")
    print(f"  {'-' * 60}")
    for city in sorted(df["city"].unique()):
        cdf = df[df["city"] == city]
        print(f"  {city:<15} {cdf['risk_score'].mean():>9.1f}% {cdf['risk_score'].max():>9.1f}% "
              f"{(cdf['risk_level'] == 'CRITICAL').sum():>14,} {(cdf['risk_level'] == 'HIGH').sum():>10,}")

    print(f"\n  Seasonal Risk Patterns:")
    seasons = {
        "Summer (Apr-Jun)": df[df["is_summer"] == 1],
        "Monsoon (Jul-Sep)": df[df["is_monsoon"] == 1],
        "Winter (Nov-Feb)": df[df["month"].isin([11, 12, 1, 2])],
    }
    for name, sdf in seasons.items():
        if len(sdf) > 0:
            print(f"    {name:<25}: avg risk {sdf['risk_score'].mean():5.1f}% | "
                  f"{(sdf['risk_level'] == 'CRITICAL').sum():,} CRITICAL hours")

    print(f"\n  Top 10 Most Dangerous Hours:")
    top10 = df.nlargest(10, "risk_score")
    for _, row in top10.iterrows():
        print(f"    {row['hour_timestamp']}  {row['city']:<12} "
              f"Temp:{row['temperature_2m']:5.1f}°C  Wind:{row['wind_speed_10m']:5.1f}km/h  "
              f"Gusts:{row['wind_gusts_10m']:5.1f}km/h  → {row['risk_score']:5.1f}% {row['risk_level']}")

    # ── 5. Save ──
    df.to_csv("data/up_predictions_v2.csv", index=False)
    size_mb = os.path.getsize("data/up_predictions_v2.csv") / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"✅ Saved to data/up_predictions_v2.csv ({size_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
