"""
Enhanced Model: More Features + Retrain
================================================================
1. Re-fetches US weather with EXPANDED variable list
2. Merges with outage data (reusing county-to-city assignments)
3. Engineers all derived features (temporal, interaction, rolling)
4. Retrains XGBoost
5. Compares old vs new model

Input:  data/us_training_data.csv (for county-to-city mapping + outage labels)
Output: data/us_training_v2.csv, models/xgboost_model_v2.json
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import requests
import time
import json
import os

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════

REPRESENTATIVE_CITIES = {
    "Texas": [
        {"name": "Houston",     "lat": 29.76, "lon": -95.37},
        {"name": "Dallas",      "lat": 32.78, "lon": -96.80},
        {"name": "San_Antonio", "lat": 29.42, "lon": -98.49},
        {"name": "El_Paso",     "lat": 31.77, "lon": -106.44},
    ],
    "Arizona": [
        {"name": "Phoenix",     "lat": 33.45, "lon": -112.07},
        {"name": "Tucson",      "lat": 32.22, "lon": -110.97},
        {"name": "Flagstaff",   "lat": 35.20, "lon": -111.65},
    ],
    "Louisiana": [
        {"name": "New_Orleans", "lat": 29.95, "lon": -90.07},
        {"name": "Baton_Rouge", "lat": 30.45, "lon": -91.19},
        {"name": "Shreveport",  "lat": 32.53, "lon": -93.75},
    ],
    "Mississippi": [
        {"name": "Jackson",     "lat": 32.30, "lon": -90.18},
        {"name": "Gulfport",    "lat": 30.37, "lon": -89.09},
        {"name": "Tupelo",      "lat": 34.26, "lon": -88.70},
    ],
    "Oklahoma": [
        {"name": "Oklahoma_City", "lat": 35.47, "lon": -97.52},
        {"name": "Tulsa",         "lat": 36.15, "lon": -95.99},
        {"name": "Lawton",        "lat": 34.60, "lon": -98.39},
    ],
    "Florida": [
        {"name": "Miami",        "lat": 25.76, "lon": -80.19},
        {"name": "Tampa",        "lat": 27.95, "lon": -82.46},
        {"name": "Jacksonville", "lat": 30.33, "lon": -81.66},
        {"name": "Pensacola",    "lat": 30.44, "lon": -87.22},
    ],
}

# EXPANDED weather variables (6 original + 4 new)
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
    # NEW variables
    "wind_gusts_10m",           # sudden gusts snap lines
    "dewpoint_2m",              # moisture/condensation damage
    "shortwave_radiation",      # solar heat load on equipment
    "weather_code",             # WMO codes: thunderstorm, fog, hail
]


# ══════════════════════════════════════════════════════════════
# 2. FETCH EXPANDED WEATHER
# ══════════════════════════════════════════════════════════════

def fetch_city_weather(city_name, lat, lon):
    """Fetch 2023 hourly weather with expanded variables."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "America/Chicago",
    }
    response = requests.get(url, params=params, timeout=60)
    if response.status_code != 200:
        print(f"  ❌ Failed for {city_name}: HTTP {response.status_code}")
        return None

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({"hour_timestamp": pd.to_datetime(hourly["time"])})
    for var in WEATHER_VARS:
        df[var] = hourly.get(var)

    df["weather_city"] = city_name
    return df


def fetch_all_weather():
    """Fetch expanded weather for all 20 cities."""
    all_weather = []
    total = sum(len(c) for c in REPRESENTATIVE_CITIES.values())
    i = 0

    print(f"Fetching expanded weather for {total} cities...")
    for state, cities in REPRESENTATIVE_CITIES.items():
        for city in cities:
            i += 1
            print(f"  [{i}/{total}] {city['name']}, {state}...", end="", flush=True)
            df = fetch_city_weather(city["name"], city["lat"], city["lon"])
            if df is not None:
                all_weather.append(df)
                print(f" ✅ {len(df)} hours, {len(WEATHER_VARS)} variables")
            else:
                print(f" ❌")
            time.sleep(1.5)

    combined = pd.concat(all_weather, ignore_index=True)
    print(f"✅ Total weather records: {len(combined):,}")
    return combined


# ══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (ALL FEATURES)
# ══════════════════════════════════════════════════════════════

def calculate_heat_index(T_celsius, RH):
    """Steadman heat index formula."""
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


def engineer_features(df):
    """Add ALL derived features to the dataframe."""
    print("Engineering features...")

    # ── Time features ──
    df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"])
    df["hour_of_day"] = df["hour_timestamp"].dt.hour
    df["day_of_week"] = df["hour_timestamp"].dt.dayofweek
    df["month"] = df["hour_timestamp"].dt.month

    # ── Flags ──
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].isin([6, 7, 8, 9, 10, 18, 19, 20, 21, 22]).astype(int)

    # ── Heat index ──
    df["heat_index"] = calculate_heat_index(df["temperature_2m"], df["relative_humidity_2m"])

    # ── NEW: Gust-to-sustained wind ratio ──
    df["gust_ratio"] = np.where(
        df["wind_speed_10m"] > 0,
        df["wind_gusts_10m"] / df["wind_speed_10m"],
        0
    )

    # ── NEW: Dew point depression (how far temp is from dew point) ──
    df["dewpoint_depression"] = df["temperature_2m"] - df["dewpoint_2m"]

    # ── NEW: Interaction terms ──
    df["temp_x_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["rain_x_wind"] = df["precipitation"] * df["wind_speed_10m"]

    # ── NEW: Is thunderstorm (from weather_code) ──
    # WMO codes 95, 96, 99 = thunderstorm; 65, 67, 75, 77 = heavy precip
    storm_codes = [95, 96, 99, 65, 67, 75, 77, 85, 86]
    df["is_thunderstorm"] = df["weather_code"].isin(storm_codes).astype(int)

    # ── NEW: Rolling averages (per weather_city group) ──
    # We sort by city + time so rolling windows are correct
    df = df.sort_values(["weather_city", "hour_timestamp"]).reset_index(drop=True)

    print("  Computing rolling averages (24h windows)...")
    for city in df["weather_city"].unique():
        mask = df["weather_city"] == city
        city_idx = df[mask].index

        # Rolling 24h average temperature (sustained heat)
        df.loc[city_idx, "rolling_avg_temp_24h"] = (
            df.loc[city_idx, "temperature_2m"].rolling(24, min_periods=1).mean()
        )

        # Rolling 24h max temperature
        df.loc[city_idx, "rolling_max_temp_24h"] = (
            df.loc[city_idx, "temperature_2m"].rolling(24, min_periods=1).max()
        )

        # Temperature change over last 3 hours
        df.loc[city_idx, "temp_change_3h"] = (
            df.loc[city_idx, "temperature_2m"].diff(3)
        )

        # Pressure change over last 3 hours (dropping pressure = storm coming)
        df.loc[city_idx, "pressure_change_3h"] = (
            df.loc[city_idx, "surface_pressure"].diff(3)
        )

        # Consecutive hot hours (above 35°C)
        hot = (df.loc[city_idx, "temperature_2m"] > 35).astype(int)
        streaks = hot.groupby((hot != hot.shift()).cumsum()).cumsum()
        df.loc[city_idx, "consecutive_hot_hours"] = streaks * hot

    print(f"  Total features engineered: {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════
# 4. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Enhanced Model Pipeline: More Features + Retrain")
    print("=" * 60)

    # ── Step 1: Load existing data for county-to-city mapping ──
    print("\n── Step 1: Loading existing training data for mappings ──")
    existing = pd.read_csv("data/us_training_data.csv",
                           usecols=["fips_code", "county", "state", "hour_timestamp",
                                    "customers_out", "outage", "weather_city"])
    print(f"  Loaded {len(existing):,} rows with county-to-city assignments")

    # ── Step 2: Fetch expanded weather ──
    print("\n── Step 2: Fetching expanded weather data ──")
    weather = fetch_all_weather()

    # ── Step 3: Merge ──
    print("\n── Step 3: Merging with outage data ──")
    existing["hour_timestamp"] = pd.to_datetime(existing["hour_timestamp"])
    merged = existing.merge(weather, on=["weather_city", "hour_timestamp"], how="inner")
    print(f"  Merged rows: {len(merged):,}")

    # ── Step 4: Engineer features ──
    print("\n── Step 4: Feature engineering ──")
    merged = engineer_features(merged)

    # Drop NaN from rolling features (first few hours per city)
    merged = merged.dropna()
    print(f"  Rows after dropping NaN: {len(merged):,}")

    # Save enhanced dataset
    merged.to_csv("data/us_training_v2.csv", index=False)
    size_mb = os.path.getsize("data/us_training_v2.csv") / (1024 * 1024)
    print(f"  Saved: data/us_training_v2.csv ({size_mb:.1f} MB)")

    # ── Step 5: Train new model ──
    print("\n── Step 5: Training enhanced model ──")

    FEATURES_V2 = [
        # Original weather
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_speed_10m", "cloud_cover", "surface_pressure",
        # New weather
        "wind_gusts_10m", "dewpoint_2m", "shortwave_radiation",
        # Time features
        "hour_of_day", "day_of_week", "month",
        # Flags
        "is_summer", "is_monsoon", "is_peak_hour", "is_thunderstorm",
        # Derived
        "heat_index", "gust_ratio", "dewpoint_depression",
        "temp_x_humidity", "rain_x_wind",
        # Temporal/rolling
        "rolling_avg_temp_24h", "rolling_max_temp_24h",
        "temp_change_3h", "pressure_change_3h",
        "consecutive_hot_hours",
    ]

    print(f"  Feature count: {len(FEATURES_V2)} (was 13)")
    X = merged[FEATURES_V2]
    y = merged["outage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.1,
        scale_pos_weight=scale_pos, random_state=42,
        n_jobs=-1, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # ── Step 6: Compare ──
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON: Original vs Enhanced")
    print("=" * 60)

    with open("models/training_metrics.json") as f:
        old = json.load(f)

    new_recall = report["1"]["recall"]
    new_precision = report["1"]["precision"]
    new_f1 = report["1"]["f1-score"]
    new_acc = report["accuracy"]

    print(f"\n  {'Metric':<20} {'Original':>10} {'Enhanced':>10} {'Change':>10}")
    print(f"  {'-' * 50}")

    def fmt_change(old_val, new_val):
        diff = (new_val - old_val) * 100
        arrow = "▲" if diff > 0 else "▼" if diff < 0 else "─"
        return f"{arrow} {abs(diff):.1f}pp"

    print(f"  {'Recall':<20} {old['outage_class_recall']:>10.3f} {new_recall:>10.3f} {fmt_change(old['outage_class_recall'], new_recall):>10}")
    print(f"  {'Precision':<20} {old['outage_class_precision']:>10.3f} {new_precision:>10.3f} {fmt_change(old['outage_class_precision'], new_precision):>10}")
    print(f"  {'F1 Score':<20} {old['outage_class_f1']:>10.3f} {new_f1:>10.3f} {fmt_change(old['outage_class_f1'], new_f1):>10}")
    print(f"  {'Accuracy':<20} {old['accuracy']:>10.3f} {new_acc:>10.3f} {fmt_change(old['accuracy'], new_acc):>10}")
    print(f"  {'Features':<20} {'13':>10} {str(len(FEATURES_V2)):>10}")

    # ── Step 7: Save if improved ──
    if new_f1 > old["outage_class_f1"]:
        print(f"\n✅ Enhanced model is BETTER! Saving...")
        model.save_model("models/xgboost_model_v2.json")

        metrics = {
            "accuracy": new_acc,
            "outage_class_recall": new_recall,
            "outage_class_precision": new_precision,
            "outage_class_f1": new_f1,
            "scale_pos_weight_used": scale_pos,
            "features": FEATURES_V2,
            "feature_count": len(FEATURES_V2),
        }
        with open("models/training_metrics_v2.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Feature importance
        importances = model.feature_importances_
        feat_imp = sorted(zip(FEATURES_V2, importances), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 Feature Importances:")
        for feat, imp in feat_imp[:10]:
            bar = "█" * int(imp * 100)
            print(f"    {feat:<25} {imp:.4f} {bar}")

        print(f"\n  Saved: models/xgboost_model_v2.json")
        print(f"  Saved: models/training_metrics_v2.json")
    else:
        print(f"\n⚠️ Enhanced model did NOT improve. Original kept.")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
