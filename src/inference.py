"""
Phase 3: Inference — Predict Outage Risk for UP/NCR Cities
================================================================
Loads the trained XGBoost model and runs it on Indian weather data
to produce outage risk predictions for each city and hour.

Input:  data/engineered_data.csv (UP/NCR weather, 6 cities, 2 years)
        models/xgboost_model.json (trained model)
Output: data/up_predictions.csv (predictions with risk scores)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os

# ── Configuration ──────────────────────────────────────────────────

INPUT_FILE = "data/engineered_data.csv"
MODEL_PATH = "models/xgboost_model.json"
OUTPUT_FILE = "data/up_predictions.csv"

# These MUST match the exact features the model was trained on, in order
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_summer",
    "is_monsoon",
    "is_peak_hour",
    "heat_index",
]


def classify_risk(probability):
    """Convert raw probability into a human-readable risk level."""
    if probability >= 0.70:
        return "CRITICAL"
    elif probability >= 0.50:
        return "HIGH"
    elif probability >= 0.30:
        return "MODERATE"
    else:
        return "LOW"


def main():
    print("=" * 60)
    print("Phase 3: Inference — UP/NCR Outage Risk Predictions")
    print("=" * 60)

    # ── Step 1: Load model ─────────────────────────────────────────
    print(f"\nLoading trained model from {MODEL_PATH}...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print("  ✅ Model loaded")

    # ── Step 2: Load Indian weather data ───────────────────────────
    print(f"\nLoading Indian weather data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows: {len(df):,}")
    print(f"  Cities: {sorted(df['city'].unique())}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── Step 3: Verify all features exist ──────────────────────────
    print(f"\nVerifying feature columns...")
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  ❌ MISSING FEATURES: {missing}")
        return
    print(f"  ✅ All 13 features present")

    # Drop rows with NaN in features
    before = len(df)
    df = df.dropna(subset=FEATURES)
    if before - len(df) > 0:
        print(f"  Dropped {before - len(df)} rows with missing values")

    # ── Step 4: Run inference ──────────────────────────────────────
    print(f"\nRunning model predictions on {len(df):,} rows...")

    X = df[FEATURES]

    # Get probability of outage (class 1)
    probabilities = model.predict_proba(X)[:, 1]

    # Add predictions to dataframe
    df["risk_score"] = (probabilities * 100).round(1)  # 0-100%
    df["risk_level"] = df["risk_score"].apply(lambda x: classify_risk(x / 100))
    df["predicted_outage"] = (probabilities >= 0.50).astype(int)

    print("  ✅ Predictions complete")

    # ── Step 5: Results summary ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    # Overall distribution
    print(f"\n  Risk Level Distribution (all cities, all hours):")
    risk_dist = df["risk_level"].value_counts()
    for level in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        count = risk_dist.get(level, 0)
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {level:>10}: {count:>8,} hours ({pct:5.1f}%) {bar}")

    # Per-city breakdown
    print(f"\n  Per-City Average Risk Score:")
    print(f"  {'City':<15} {'Avg Risk':>10} {'Peak Risk':>10} {'CRITICAL hrs':>14} {'HIGH hrs':>10}")
    print(f"  {'-'*60}")
    for city in sorted(df["city"].unique()):
        city_df = df[df["city"] == city]
        avg_risk = city_df["risk_score"].mean()
        peak_risk = city_df["risk_score"].max()
        critical = (city_df["risk_level"] == "CRITICAL").sum()
        high = (city_df["risk_level"] == "HIGH").sum()
        print(f"  {city:<15} {avg_risk:>9.1f}% {peak_risk:>9.1f}% {critical:>14,} {high:>10,}")

    # Seasonal breakdown
    print(f"\n  Seasonal Risk Patterns:")
    seasons = {
        "Summer (Apr-Jun)": df[df["is_summer"] == 1],
        "Monsoon (Jul-Sep)": df[df["is_monsoon"] == 1],
        "Winter (Nov-Feb)": df[df["month"].isin([11, 12, 1, 2])],
        "Post-Monsoon (Oct)": df[df["month"] == 10],
    }
    for name, season_df in seasons.items():
        if len(season_df) > 0:
            avg = season_df["risk_score"].mean()
            crit = (season_df["risk_level"] == "CRITICAL").sum()
            print(f"    {name:<25}: avg risk {avg:5.1f}% | {crit:,} CRITICAL hours")

    # Top 10 most dangerous hours
    print(f"\n  Top 10 Most Dangerous Hours (Highest Risk Score):")
    top10 = df.nlargest(10, "risk_score")[["timestamp", "city", "temperature_2m",
                                           "wind_speed_10m", "precipitation",
                                           "risk_score", "risk_level"]]
    for _, row in top10.iterrows():
        print(f"    {row['timestamp']}  {row['city']:<12} "
              f"Temp:{row['temperature_2m']:5.1f}°C  Wind:{row['wind_speed_10m']:5.1f}km/h  "
              f"Rain:{row['precipitation']:5.1f}mm  → {row['risk_score']:5.1f}% {row['risk_level']}")

    # ── Step 6: Save ───────────────────────────────────────────────
    print(f"\nSaving predictions to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"✅ DONE! Saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"   {len(df):,} hourly predictions for {df['city'].nunique()} cities")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
