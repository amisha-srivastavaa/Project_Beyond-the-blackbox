"""
Step 3: Feature Engineering
============================
This script reads the raw weather data and adds smarter columns
that help the AI model spot patterns more easily.

Input:  weather_data.csv       (raw weather, 8 columns)
Output: engineered_data.csv    (enriched, 15 columns)
"""

import pandas as pd
import numpy as np  # For math operations (heat index formula)

# ── 1. Load the raw weather data ────────────────────────────────────────

print("Loading weather_data.csv...")
df = pd.read_csv("weather_data.csv")
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

# ── 2. Parse the timestamp into a proper datetime object ────────────────
# Right now "timestamp" is just text like "2023-01-01T00:00".
# We convert it so Python understands it as an actual date and time.

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── 3. Extract time-based features ─────────────────────────────────────
# These tell the model WHEN something happened, which matters a lot.
# Power demand is very different at 3 AM vs 7 PM.

df["hour_of_day"] = df["timestamp"].dt.hour        # 0 to 23
df["day_of_week"] = df["timestamp"].dt.dayofweek    # 0=Monday, 6=Sunday
df["month"] = df["timestamp"].dt.month              # 1 to 12

print("  ✅ Added: hour_of_day, day_of_week, month")

# ── 4. Create season flags ──────────────────────────────────────────────
# In UP, summer (Apr-Jun) means transformer overload risk.
# Monsoon (Jul-Sep) means line fault risk from rain and wind.
# These binary flags (0 or 1) make it easy for the model to learn this.

df["is_summer"] = df["month"].apply(lambda m: 1 if m in [4, 5, 6] else 0)
df["is_monsoon"] = df["month"].apply(lambda m: 1 if m in [7, 8, 9] else 0)

print("  ✅ Added: is_summer, is_monsoon")

# ── 5. Create peak hour flag ───────────────────────────────────────────
# Peak electricity demand in India is typically:
#   Morning: 6 AM to 10 AM (everyone waking up, ACs/geysers turning on)
#   Evening: 6 PM to 10 PM (lights, TVs, cooking, ACs)
# During peak hours, the grid is under maximum stress.

df["is_peak_hour"] = df["hour_of_day"].apply(
    lambda h: 1 if (6 <= h <= 10) or (18 <= h <= 22) else 0
)

print("  ✅ Added: is_peak_hour")

# ── 6. Calculate Heat Index ────────────────────────────────────────────
# Heat Index combines temperature and humidity into a single number
# that represents how hot it actually FEELS. This matters because
# transformers fail based on real thermal stress, not just air temperature.
#
# We use the simplified Steadman formula (commonly used in meteorology).
# It works for temperatures above 27°C. Below that, we just use temperature.

def calculate_heat_index(temp_c, humidity):
    """Calculate heat index from temperature (°C) and relative humidity (%)."""
    if temp_c < 27:
        return temp_c  # Heat index only meaningful in warm conditions

    # Convert to Fahrenheit for the standard formula
    temp_f = (temp_c * 9 / 5) + 32

    # Steadman's regression equation
    hi_f = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * humidity
        - 0.22475541 * temp_f * humidity
        - 0.00683783 * temp_f ** 2
        - 0.05481717 * humidity ** 2
        + 0.00122874 * temp_f ** 2 * humidity
        + 0.00085282 * temp_f * humidity ** 2
        - 0.00000199 * temp_f ** 2 * humidity ** 2
    )

    # Convert back to Celsius
    hi_c = (hi_f - 32) * 5 / 9
    return round(hi_c, 2)

df["heat_index"] = df.apply(
    lambda row: calculate_heat_index(row["temperature_2m"], row["relative_humidity_2m"]),
    axis=1
)

print("  ✅ Added: heat_index")

# ── 7. Assign a numeric region ID to each city ─────────────────────────
# ML models work with numbers, not text. So we convert city names
# to numeric codes. This is called "label encoding".

city_to_id = {
    "Lucknow": 0,
    "Noida": 1,
    "Ghaziabad": 2,
    "Agra": 3,
    "Firozabad": 4,
    "Meerut": 5,
}
df["region_id"] = df["city"].map(city_to_id)

print("  ✅ Added: region_id")

# ── 8. Save the enriched dataset ───────────────────────────────────────

output_file = "engineered_data.csv"
df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print(f"✅ Done! Saved {len(df)} rows with {len(df.columns)} columns to '{output_file}'")
print(f"\nColumns in the final dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print(f"\nSample row (Firozabad, a summer afternoon):")
sample = df[(df["city"] == "Firozabad") & (df["month"] == 5) & (df["hour_of_day"] == 15)]
if len(sample) > 0:
    print(sample.iloc[0].to_string())
print("=" * 60)

if __name__ == "__main__":
    pass  # All code runs at module level for simplicity
