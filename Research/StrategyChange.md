# Strategy Update — Data & Class Imbalance Approach

## ⚠️ Outage Label Strategy — CHANGED

### What We Are NOT Doing Anymore
Do NOT generate outage labels using rule-based thresholds on weather data
(e.g. "if temperature > 40°C → outage = 1"). This approach has been abandoned.
It is not aligned with research literature and produces an unreliable model.

### What We Are Doing Instead

**Step 1 — EAGLE-I Dataset (Real US Outage Data)**
Use the US Department of Energy EAGLE-I dataset as the source of outage labels.
It contains real, hourly, county-level outage events across the US (2014–2022).
This is the same dataset used in the research papers this project is based on.

**Step 2 — Match with NOAA Weather Data**
Match EAGLE-I outage records to NOAA weather station data by timestamp and location.
This produces a merged dataset of real weather conditions paired with real outage events.

**Step 3 — Train on US Data, Apply to UP**
The weather-to-grid-failure relationship is physically universal — the same weather
conditions that cause grid failures in the US cause them in UP. Train the model on the
merged US dataset, then run inference on our existing `weather_data.csv` (UP/NCR cities).
Known limitation: UP's grid is older and less resilient, so the model will be slightly
conservative. This is documented and explainable.

---

## Class Imbalance Strategy

Outages are rare. Without correction the model will always predict "no outage" and
appear accurate while being useless. Two complementary fixes are applied:

**SMOGN — applied before training (data-level fix)**
Generates synthetic minority-class samples by interpolating between real outage rows.
This is NOT the same as fabricating labels — it only creates new examples from real ones.
Library: `smogn`

**Cost-Sensitive Learning — applied during training (model-level fix)**
XGBoost parameter: `scale_pos_weight = (no-outage row count) / (outage row count)`
Penalises the model for missing real outages, preventing it from defaulting to "no outage".

Both are used together for maximum robustness.
