# AI-Powered Power Outage Predictor (UP/NCR Region)
## Comprehensive Project Analysis & Documentation

---

## 1. The Core Idea & Objective
**Objective:** To build an AI-powered application for the Uttar Pradesh/National Capital Region (UP/NCR) capable of accurately predicting unscheduled, weather-induced power outages.

**The Problem:** Traditional outage prediction often relies on simple weather thresholds (e.g., "if wind > 60 km/h, predict outage"). This approach fails to capture complex, non-linear relationships, such as how prolonged high humidity combined with moderate heat degrades transformer lifespans over time. 

**The Solution:** An XGBoost-based Machine Learning model that empirically learns the complex relationships between granular weather variables and physical grid failure.

**Target Cities:** Lucknow, Noida, Ghaziabad, Agra, Meerut, Firozabad.

---

## 2. Research Foundation

Four academic papers were reviewed to inform the project's architecture:

### Paper 1 (PRIMARY BLUEPRINT)
**"Machine Learning Model Development to Predict Power Outage Duration"** — Ghasemkhani et al. (2024), Sensors Journal.
- Used **XGBoost + MRMR feature selection** → achieved **97.66% accuracy**
- Justified our choice of XGBoost, MRMR, and classification framing
- Key finding: XGBoost with MRMR consistently outperformed other algorithms across all seasons

### Paper 2
**"Deep Learning-Based Weather-Related Power Outage Prediction with Socio-Economic and Power Infrastructure Data"** — Wang et al. (Wayne State University, 2024)
- Used **household income, power infrastructure density (OpenStreetMap), weather station distances** as additional features
- Conditional MLP architecture with socio-economic conditioning
- **Key insight:** Same weather causes different outage rates in wealthy vs. poor neighborhoods → inspired our infrastructure fragility multiplier

### Paper 3
- Explored LSTM-based temporal models for sequential outage prediction

### Paper 4 (Graph Neural Networks)
- Used GNNs to model spatial cascade of outages (substation → substation)
- Cited as "future scope"

---

## 3. The Strategy Pivot: Cross-Continental Transfer Learning
Initially, the project considered generating "synthetic outage labels" based on predefined weather rules for the Indian dataset. This approach was abandoned to ensure scientific defensibility — it would be circular reasoning where the model only learns the rules we hardcoded.

**The New Paradigm:**
Because the physics of grid failure—transformers overloading under thermal stress or transmission lines snapping under wind loads—are universal, the model was trained on **real, historical US outage data** and evaluated on Indian weather data. 

```
US Weather + US Outage Data → XGBoost Training → Trained Model
Indian Weather Data → Trained Model → Indian Risk Scores
```

---

## 4. Data Sourcing
1. **Power Outage Data (Target):** The EAGLE-I Historic Dataset (2023) from Oak Ridge National Laboratory (ORNL) / US Dept. of Energy. 26 million rows of county-level outage data at 15-minute intervals.
2. **Weather Data (Features):** Open-Meteo Archive API (ERA5 Reanalysis). Hourly historical weather for any global coordinate.

---

## 5. Data Filtering & Engineering Pipeline

### Part A: Filtering EAGLE-I Data
**Script:** `filter_eagle_i.py`

1. **Input:** 26M rows (1.19 GB), all US counties
2. **State Selection:** Filtered to 6 US states mimicking UP/NCR climate:
   - **Texas & Arizona:** Extreme summer heat (~45°C) → simulating UP summer
   - **Louisiana & Florida:** High humidity + tropical storms → simulating monsoon
   - **Mississippi & Oklahoma:** Thunderstorms, rural grid characteristics
3. **Temporal Aggregation:** 15-minute intervals → hourly records
4. **Outage Threshold:** `customers_out > 100` = outage (1), otherwise normal (0)
   - Rationale: <100 customers = routine maintenance, not grid failure
5. **Output:** 1.7M rows, 556 counties (`eagle_i_filtered.csv`, 77 MB)

### Part B: Weather Matching
**Script:** `fetch_us_weather.py`

1. Downloaded county centroid coordinates (lat/lon) from GitHub/Census Bureau
2. Defined 20 representative cities across 6 states
3. Used haversine distance to assign each county to nearest city
4. Fetched 2023 hourly weather from Open-Meteo for all 20 cities
5. Inner-joined outage data with weather on `(weather_city, hour_timestamp)`
6. **Output:** `us_training_data.csv` (135 MB, ~1.7M rows)

### Part C: Feature Engineering (v1 — 13 features)
**Script:** `engineer_us_features.py`

| Feature | Type | Logic |
|---|---|---|
| `hour_of_day` | Time | 0-23 |
| `day_of_week` | Time | 0-6 (Mon-Sun) |
| `month` | Time | 1-12 |
| `is_summer` | Flag | Jun-Aug (US) / Apr-Jun (India) |
| `is_monsoon` | Flag | Jul-Sep |
| `is_peak_hour` | Flag | 6-10 AM, 6-10 PM |
| `heat_index` | Derived | Steadman's regression formula — "feels like" thermal stress |

**Output:** `us_training_final.csv` (186 MB, 13 features + target)

### Indian Weather Data (Parallel)
**Scripts:** `fetch_weather.py`, `feature_engineering.py`
- 2 years (2023-2024) hourly weather for 6 Indian cities
- Same feature engineering with Indian seasonal definitions
- **Output:** `engineered_data.csv` (7.7 MB, 105,264 rows)

---

## 6. Model Training: v1 (13 Features)
**Script:** `train_model.py`

### Algorithm: XGBoost Classifier
Excels at tabular data, handles non-linear interactions, fast and interpretable.

### Handling Class Imbalance
Outages occurred only **13.6%** of the time. A naive "always predict normal" model = 86.4% accuracy but 0% usefulness.

**Approach: Cost-Sensitive Learning** — `scale_pos_weight = 6.37`
- Model penalized 6.37x more for missing an outage than for a false alarm
- Chosen over SMOTE to avoid creating artificial weather profiles

### v1 Hyperparameters
`n_estimators=200, max_depth=6, learning_rate=0.1`

### v1 Results (threshold = 0.50)

| Metric | Value |
|---|---|
| Accuracy | 64.6% |
| Outage Recall | 62.7% |
| Outage Precision | 21.9% |
| F1 Score | 0.325 |

### v1 Feature Importance
1. `is_summer` — Summer = most dangerous season
2. `heat_index` — Thermal stress > raw temperature
3. `month` — Seasonal patterns
4. `relative_humidity` — Amplifies heat damage
5. `surface_pressure` — Low pressure = storms

**Artifacts:** `models/xgboost_model.json`, `models/training_metrics.json`, `models/feature_importance.png`

---

## 7. Model Enhancement: v2 (26 Features)
**Script:** `enhance_and_retrain.py`

### New Weather Variables (4 additional from Open-Meteo)
| Variable | Why |
|---|---|
| `wind_gusts_10m` | Sudden gusts snap lines — more dangerous than sustained wind |
| `dewpoint_2m` | Moisture/condensation damage |
| `shortwave_radiation` | Solar heat load on transformers |
| `weather_code` | WMO codes for thunderstorms, hail, fog |

### New Derived Features (9 additional)
| Feature | What It Captures |
|---|---|
| `gust_ratio` | wind_gusts / wind_speed — gustiness |
| `dewpoint_depression` | temp − dewpoint — condensation risk |
| `temp_x_humidity` | temperature × humidity — combined thermal stress |
| `rain_x_wind` | precipitation × wind — storm intensity |
| `is_thunderstorm` | Flag from WMO weather codes |
| `rolling_avg_temp_24h` | 24-hour rolling average temperature — sustained heat |
| `rolling_max_temp_24h` | 24-hour rolling max — peak stress |
| `temp_change_3h` | Rapid temperature swings stress equipment |
| `pressure_change_3h` | Rapid pressure drops = storm arrival |
| `consecutive_hot_hours` | Hours in row above 35°C — cumulative damage |

### v2 Hyperparameters
`n_estimators=300, max_depth=7, learning_rate=0.1`

### v2 Results (threshold = 0.50)

| Metric | v1 (13 feat) | v2 (26 feat) | Change |
|---|---|---|---|
| Accuracy | 64.6% | **67.3%** | ▲ +2.7pp |
| Recall | 62.7% | **62.9%** | ▲ +0.2pp |
| Precision | 21.9% | **23.5%** | ▲ +1.6pp |
| F1 Score | 0.325 | **0.343** | ▲ +1.8pp |

### v2 Feature Importance
1. **`temp_x_humidity`** (NEW) — #1 most important. Combined thermal-moisture stress.
2. `is_summer`
3. `month`
4. `surface_pressure`
5. `is_monsoon`
6. **`rolling_max_temp_24h`** (NEW) — Sustained heat > momentary spikes
7. `day_of_week`
8. **`dewpoint_2m`** (NEW) — Moisture damage signal
9. **`wind_gusts_10m`** (NEW) — Gusts snap lines
10. `hour_of_day`

**4 of the top 10 features are NEW** — confirming genuine predictive value.

**Artifacts:** `models/xgboost_model_v2.json`, `models/training_metrics_v2.json`

---

## 8. MRMR Feature Selection
**Script:** `mrmr_selection.py`

### What is MRMR?
**Minimum Redundancy Maximum Relevance** — selects features that are strongly correlated with the target while removing features that overlap with each other. Used in Paper 1 to achieve 97%+ accuracy.

### Results
MRMR ranked all 13 v1 features. Key finding: `temperature_2m` was ranked 9th (low) because `heat_index` already captures that information — **redundancy correctly detected**.

### Feature Subset Testing

| K (features kept) | Recall | Precision | F1 |
|---|---|---|---|
| K=6 | 0.598 | 0.201 | 0.301 |
| K=8 | 0.630 | 0.213 | 0.318 |
| K=10 | 0.630 | 0.216 | 0.322 |
| K=13 (all) | 0.627 | 0.219 | 0.325 |

### Conclusion
MRMR did NOT improve performance — with only 13 features, there wasn't enough redundancy to trim. The ranking validates that all features contribute meaningfully. Saved for documentation.

**Artifact:** `models/mrmr_features.json`

---

## 9. Threshold Optimization

### The Problem
Default threshold 0.50 is arbitrary. The model outputs probabilities (0-100%) — we tested which cutoff point maximizes the F1 score.

### Results (v2 model)

| Threshold | Recall | Precision | F1 | Accuracy | False Alarms |
|---|---|---|---|---|---|
| 0.30 | 94.3% | 15.5% | 0.266 | 29.5% | 240,995 |
| 0.40 | 81.9% | 18.6% | 0.304 | 49.0% | 167,545 |
| 0.50 (default) | 62.9% | 23.5% | 0.343 | 67.3% | 95,746 |
| **0.55 (optimal)** | **51.6%** | **27.0%** | **0.354** | **74.4%** | **65,600** |
| 0.60 | 40.0% | 31.4% | 0.352 | 80.0% | 41,016 |
| 0.70 | 19.0% | 42.3% | 0.262 | 85.5% | 12,111 |

### Final Model Configuration (Optimal)

| Metric | Value |
|---|---|
| **Accuracy** | **74.4%** |
| **Recall** | **51.6%** |
| **Precision** | **27.0%** |
| **F1 Score** | **0.354** |
| Features | 26 |
| Threshold | 0.55 |

**Artifacts:** `models/model_config.json`, `models/threshold_analysis.png`

---

## 10. Phase 3: Inference on Indian Data
**Scripts:** `inference.py` (v1), `inference_v2.py` (v2)

### Process
1. Re-fetched 2 years of hourly weather for 6 Indian cities with all 10 weather variables
2. Engineered all 26 features (Indian seasons: summer = Apr-Jun)
3. Fed through `xgboost_model_v2.json`
4. Generated probability scores → risk levels

### Risk Level Definitions
| Level | Probability | Meaning |
|---|---|---|
| LOW | < 30% | Grid is safe |
| MODERATE | 30-50% | Elevated caution |
| HIGH | 50-70% | Significant outage risk |
| CRITICAL | ≥ 70% | Grid failure highly likely |

### Results (105,246 predictions, 6 cities, 2 years)

**Risk Distribution:**
| Level | Hours | % of Time |
|---|---|---|
| CRITICAL | 3,787 | 3.6% |
| HIGH | 11,606 | 11.0% |
| MODERATE | 41,062 | 39.0% |
| LOW | 48,791 | 46.4% |

**Per-City Results:**
| City | Avg Risk | Peak Risk | CRITICAL Hours |
|---|---|---|---|
| Meerut | 36.7% | 96.2% | 957 |
| Ghaziabad | 35.4% | 94.9% | 940 |
| Noida | 34.5% | 95.7% | 777 |
| Agra | 33.6% | 94.5% | 565 |
| Firozabad | 33.0% | 94.1% | 430 |
| Lucknow | 32.9% | 94.7% | 118 |

**Seasonal Patterns:**
| Season | Avg Risk | CRITICAL Hours |
|---|---|---|
| **Summer (Apr-Jun)** | **45.3%** | **3,599** |
| Monsoon (Jul-Sep) | 36.4% | 154 |
| Winter (Nov-Feb) | 29.0% | 27 |

**Most Dangerous Hour:** May 27, 2024, 2:00 PM, Meerut — 46.0°C, 37 km/h gusts → **96.2% CRITICAL**

**Artifact:** `data/up_predictions_v2.csv` (22 MB)

---

## 11. Infrastructure Fragility Scores
Inspired by Paper 2: same weather causes different outage rates depending on local infrastructure quality.

### Approach 1: OpenStreetMap (Attempted, Failed)
Queried OSM Overpass API for power infrastructure per city. **Problem:** Data was misleading — Firozabad showed highest density because transmission lines pass through it (serving other cities). OSM mapping quality varies wildly between Indian cities.

### Approach 2: DISCOM Performance Data (Adopted)
Used **official government data** from UPERC/PFC FY 2023-24:

| DISCOM | Distribution Losses | Rating | Cities |
|---|---|---|---|
| **PVVNL** | **13.44%** (best) | **A+** | Noida, Ghaziabad, Meerut |
| **MVVNL** | 15.23% | B- | Lucknow |
| **DVVNL** | **17.10%** (worst) | B- | Agra, Firozabad |

### Formula
`Fragility = (DISCOM_loss + city_adjustment) / baseline (PVVNL = 13.44%)`

### Final Scores
| City | DISCOM | Fragility | Grid Quality |
|---|---|---|---|
| **Noida** | PVVNL (A+) | **0.93** | Strong — planned NCR city |
| **Ghaziabad** | PVVNL (A+) | **1.00** | Good — benchmark |
| **Meerut** | PVVNL (A+) | **1.07** | Average |
| **Lucknow** | MVVNL (B-) | **1.13** | Average |
| **Agra** | DVVNL (B-) | **1.27** | Below average |
| **Firozabad** | DVVNL (B-) | **1.40** | Most fragile |

### Impact (same 45% raw risk):
- Noida: 45% × 0.93 = **41.9%**
- Firozabad: 45% × 1.40 = **63.0%**

**Artifact:** `models/fragility_scores.json`

---

## 12. Real-World Testing

### Test 1 (April 16, 2026, 6:16 PM IST) — All Cities
| City | Temp | Risk | Result |
|---|---|---|---|
| Lucknow | 35.8°C, dry | 30.4% MODERATE | No outage ✅ |
| Noida | 35.4°C, dry | 44.6% MODERATE | No outage (arguably over-predicted) |
| Ghaziabad | 35.4°C, dry | 44.6% MODERATE | No outage (arguably over-predicted) |

### Test 2 (April 17, 2026, 6:37 PM IST) — Ghaziabad
- **Conditions:** 33.1°C, 39% humidity, light rain, 100% cloud cover
- **Model prediction:** 35.9% MODERATE
- **Reality:** Actual power outage occurred during light rain

### Critical Finding
The model gave **higher risk yesterday** (44.6%, no outage, dry heat) than **today** (35.9%, actual outage, rain). This reveals the fundamental limitation: the model was trained on US failure patterns where **heat is the primary driver**. In India, **rain + wind** causes more outages because Indian infrastructure isn't weatherproofed like US equipment.

---

## 13. Known Limitations

1. **Cross-Continental Transfer Gap:** Model learned US grid failure patterns (heat-dominated). Indian grid fails differently (rain, overloaded transformers, poor maintenance). Model underweights rain/wind risk.

2. **Weather-Only Features:** Paper 1 achieved 97% with utility-specific data (equipment age, maintenance records). We only have weather — half the picture.

3. **No Rolling Features in Live Mode:** Real-time predictions approximate `rolling_avg_temp_24h` with current temperature (no 24-hour cache).

4. **Class Imbalance:** Only 13.6% of training data = outages. Fundamentally limits precision.

5. **Precision vs. Recall Trade-off:** Intentionally tuned for high recall (catching outages) at the cost of precision (false alarms). This is correct for a safety tool but means many warnings don't result in outages.

---

## 14. File Structure

```
NSUT Project/
├── data/
│   ├── outage_data_2023.csv           ← Raw EAGLE-I (1.19 GB)
│   ├── eagle_i_filtered.csv           ← Filtered 6 states (77 MB)
│   ├── us_training_data.csv           ← Outage + weather merged (135 MB)
│   ├── us_training_final.csv          ← v1 features (186 MB)
│   ├── us_training_v2.csv             ← v2 features, 26 cols (354 MB)
│   ├── weather_data.csv               ← UP/NCR raw weather (5 MB)
│   ├── engineered_data.csv            ← UP/NCR v1 features (7.7 MB)
│   ├── up_predictions.csv             ← v1 predictions (9 MB)
│   ├── up_predictions_v2.csv          ← v2 predictions (22 MB)
│   ├── dvvnl_power_map.xlsx           ← DVVNL utility reference
│   └── dvvnl_roaster.pdf              ← DVVNL utility reference
│
├── models/
│   ├── xgboost_model.json             ← v1 model (1.4 MB)
│   ├── xgboost_model_v2.json          ← v2 model (3.8 MB)
│   ├── training_metrics.json          ← v1 metrics
│   ├── training_metrics_v2.json       ← v2 metrics
│   ├── model_config.json              ← Threshold config (0.55)
│   ├── mrmr_features.json             ← MRMR ranking
│   ├── fragility_scores.json          ← DISCOM-based city scores
│   ├── feature_importance.png         ← v1 importance chart
│   └── threshold_analysis.png         ← Threshold optimization chart
│
├── filter_eagle_i.py                  ← Part A: Filter EAGLE-I
├── fetch_us_weather.py                ← Part B: Fetch US weather + merge
├── engineer_us_features.py            ← Part C: Feature engineering (US)
├── enhance_and_retrain.py             ← v2: More features + retrain
├── mrmr_selection.py                  ← MRMR feature selection
├── train_model.py                     ← Train XGBoost
├── inference.py                       ← v1 inference
├── inference_v2.py                    ← v2 inference
├── fetch_weather.py                   ← Fetch UP/NCR weather
├── feature_engineering.py             ← Feature engineering (India)
├── requirements.txt                   ← Python dependencies
├── Research/                          ← Papers + documentation
└── venv/                              ← Python environment
```

---

## 15. What Remains

### Immediate
| Task | Status |
|---|---|
| Rain hazard adjustment layer for Indian conditions | ❌ |
| Integrate fragility multipliers into predictions | ❌ |
| Backend API (FastAPI) | ❌ |
| Frontend Dashboard (risk map + charts) | ❌ |
| Real-time forecast mode (next 24-48 hours) | ❌ |

### Future Scope
- Train on Indian outage data (if DISCOMs publish records)
- Graph Neural Networks for cascade modeling (Paper 4)
- LSTM temporal models (Paper 3)
- Full socio-economic integration (Paper 2)
- Push notification system for CRITICAL alerts
