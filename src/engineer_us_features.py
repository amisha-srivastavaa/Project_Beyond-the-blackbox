"""
Part C: Feature Engineering on US Training Data
================================================================
Reads the merged US training dataset and adds derived features
necessary for the XGBoost model to learn weather-grid failure patterns.

Input:  data/us_training_data.csv
Output: data/us_training_final.csv
"""

import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/us_training_data.csv"
OUTPUT_FILE = "data/us_training_final.csv"

def calculate_heat_index(T_celsius, RH):
    """
    Calculate heat index using the Rothfusz regression (Steadman's formula).
    Input: T in Celsius, RH in % (0-100)
    Output: Heat Index in Celsius
    """
    # Convert Celsius to Fahrenheit for the standard formula
    T = (T_celsius * 9/5) + 32
    
    # Simple unadjusted calculation
    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    
    # If the simple HI is 80 or higher, use the full regression equation
    # We apply this element-wise using np.where
    
    full_hi = (-42.379 + 2.04901523*T + 10.14333127*RH - .22475541*T*RH - 
               .00683783*T*T - .05481717*RH*RH + .00122874*T*T*RH + 
               .00085282*T*RH*RH - .00000199*T*T*RH*RH)
               
    # Adjustments
    adj1 = ((13 - RH) / 4) * np.sqrt((17 - np.abs(T - 95.)) / 17)
    adj2 = ((RH - 85) / 10) * ((87 - T) / 5)
    
    # Apply adjustments where conditions are met
    full_hi = np.where((RH < 13) & (T >= 80) & (T <= 112), full_hi - adj1, full_hi)
    full_hi = np.where((RH > 85) & (T >= 80) & (T <= 87), full_hi + adj2, full_hi)
    
    # Choose between simple and full based on simple HI value
    HI_final_F = np.where(HI >= 80, full_hi, HI)
    
    # Sometimes in extreme cold, HI formula goes crazy, fallback to original temp
    HI_final_F = np.where(T < 40, T, HI_final_F)
    
    # Convert back to Celsius
    HI_final_C = (HI_final_F - 32) * 5/9
    return HI_final_C

def main():
    print("=" * 60)
    print("Part C: US Data Feature Engineering")
    print("=" * 60)

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print("Converting timestamp...")
    df['hour_timestamp'] = pd.to_datetime(df['hour_timestamp'])
    
    print("Extracting time features...")
    df['hour_of_day'] = df['hour_timestamp'].dt.hour
    df['day_of_week'] = df['hour_timestamp'].dt.dayofweek
    df['month'] = df['hour_timestamp'].dt.month
    
    print("Creating seasonal and peak flags...")
    # US Summer: June (6) to August (8)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    
    # US Hurricane/Storm Season peaking around July-September
    df['is_monsoon'] = df['month'].isin([7, 8, 9]).astype(int)
    
    # Peak demand hours: 6-10 AM (6-10) and 6-10 PM (18-22)
    df['is_peak_hour'] = df['hour_of_day'].isin([6, 7, 8, 9, 10, 18, 19, 20, 21, 22]).astype(int)
    
    print("Calculating Heat Index...")
    df['heat_index'] = calculate_heat_index(df['temperature_2m'], df['relative_humidity_2m'])
    
    print("Validating new features...")
    print(f"  Rows: {len(df):,}")
    print(f"  New columns added: hour_of_day, day_of_week, month, is_summer, is_monsoon, is_peak_hour, heat_index")
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print("=" * 60)
    print(f"✅ DONE! Saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
