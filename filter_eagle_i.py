"""
Part A: Filter EAGLE-I Outage Data
====================================
Reads the raw 26M-row EAGLE-I 2023 dataset and produces a clean,
filtered, hourly-aggregated file for only our 6 target US states.

Input:  data/outage_data_2023.csv       (26M rows, ~1.19 GB)
Output: data/eagle_i_filtered.csv       (much smaller, hourly, 6 states only)

Target states: Texas, Arizona, Louisiana, Mississippi, Oklahoma, Florida
Outage threshold: sum > 100 customers = outage (1), else (0)
"""

import pandas as pd
import os

# ── 1. Configuration ───────────────────────────────────────────────────

TARGET_STATES = [
    "Texas",
    "Arizona",
    "Louisiana",
    "Mississippi",
    "Oklahoma",
    "Florida",
]

OUTAGE_THRESHOLD = 100  # customers without power
INPUT_FILE = "data/outage_data_2023.csv"
OUTPUT_FILE = "data/eagle_i_filtered.csv"
CHUNK_SIZE = 500_000  # read 500K rows at a time to manage memory

# ── 2. Read and filter in chunks ───────────────────────────────────────

print("=" * 60)
print("EAGLE-I Data Filtering Pipeline")
print(f"Input:      {INPUT_FILE}")
print(f"States:     {', '.join(TARGET_STATES)}")
print(f"Threshold:  sum > {OUTAGE_THRESHOLD} → outage = 1")
print("=" * 60)

# We collect filtered chunks here
filtered_chunks = []
total_rows_read = 0
total_rows_kept = 0
chunk_num = 0

print("\nReading and filtering in chunks...")

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    chunk_num += 1
    total_rows_read += len(chunk)

    # Filter to only our 6 target states (exact match)
    mask = chunk["state"].isin(TARGET_STATES)
    filtered = chunk[mask].copy()
    total_rows_kept += len(filtered)

    if len(filtered) > 0:
        filtered_chunks.append(filtered)

    if chunk_num % 10 == 0:
        print(f"  Processed {total_rows_read:>12,} rows | kept {total_rows_kept:>10,}")

print(f"  Processed {total_rows_read:>12,} rows | kept {total_rows_kept:>10,}")
print(f"\n✅ Filtering complete.")
print(f"   Kept {total_rows_kept:,} of {total_rows_read:,} rows ({total_rows_kept/total_rows_read*100:.1f}%)")

# ── 3. Combine all filtered chunks ────────────────────────────────────

print("\nCombining filtered chunks...")
df = pd.concat(filtered_chunks, ignore_index=True)

# Verify: no rows from other states leaked through
states_present = sorted(df["state"].unique())
print(f"  States present after filter: {states_present}")
assert set(states_present).issubset(set(TARGET_STATES)), \
    f"ERROR: Non-target states found: {set(states_present) - set(TARGET_STATES)}"
print("  ✅ State filter verified — only target states present")

# ── 4. Parse timestamps and aggregate to hourly ───────────────────────

print("\nParsing timestamps...")
df["run_start_time"] = pd.to_datetime(df["run_start_time"])

# Floor to the nearest hour (e.g., 14:15 → 14:00, 14:30 → 14:00)
df["hour_timestamp"] = df["run_start_time"].dt.floor("h")

print("Aggregating from 15-minute to hourly intervals...")
print("  Method: max customers_out per county per hour")

# Group by county (fips_code) + hour, take the MAX outage count
# We also keep county name and state for readability
hourly = df.groupby(["fips_code", "county", "state", "hour_timestamp"], as_index=False).agg(
    customers_out=("sum", "max")  # worst moment in each hour
)

print(f"  Rows before aggregation: {len(df):,}")
print(f"  Rows after aggregation:  {len(hourly):,}")
print(f"  Compression ratio: {len(df)/len(hourly):.1f}x")

# ── 5. Create binary outage label ─────────────────────────────────────

print(f"\nCreating outage labels (threshold: > {OUTAGE_THRESHOLD} customers)...")
hourly["outage"] = (hourly["customers_out"] > OUTAGE_THRESHOLD).astype(int)

outage_count = hourly["outage"].sum()
no_outage_count = len(hourly) - outage_count
outage_rate = outage_count / len(hourly) * 100

print(f"  Outage (1):    {outage_count:>10,} rows ({outage_rate:.1f}%)")
print(f"  No outage (0): {no_outage_count:>10,} rows ({100-outage_rate:.1f}%)")

# ── 6. Per-state validation ──────────────────────────────────────────

print("\n=== PER-STATE BREAKDOWN ===")
print(f"{'State':<15} {'Rows':>10} {'Counties':>10} {'Outage %':>10}")
print("-" * 50)
for state in TARGET_STATES:
    state_df = hourly[hourly["state"] == state]
    counties = state_df["fips_code"].nunique()
    rate = state_df["outage"].mean() * 100
    print(f"{state:<15} {len(state_df):>10,} {counties:>10} {rate:>9.1f}%")

# ── 7. Sort and save ─────────────────────────────────────────────────

print(f"\nSorting by state → county → time...")
hourly = hourly.sort_values(["state", "fips_code", "hour_timestamp"]).reset_index(drop=True)

print(f"Saving to {OUTPUT_FILE}...")
hourly.to_csv(OUTPUT_FILE, index=False)

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"\n{'=' * 60}")
print(f"✅ DONE!")
print(f"   Output file: {OUTPUT_FILE}")
print(f"   File size:   {file_size_mb:.1f} MB")
print(f"   Total rows:  {len(hourly):,}")
print(f"   Columns:     {list(hourly.columns)}")
print(f"\n   Sample rows:")
print(hourly.head(3).to_string(index=False))
print(f"{'=' * 60}")
