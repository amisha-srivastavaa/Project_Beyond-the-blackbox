# ⚡ Beyond the Black Box — AI-Powered Power Outage Predictor

> An evidence-based, explainable ML system for predicting weather-induced power outages in India's UP/NCR region, trained on real US outage data via cross-continental transfer learning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#)

---

## 📌 Problem Statement

Traditional outage prediction relies on simplistic weather thresholds (e.g., *"if wind > 60 km/h → predict outage"*). This fails to capture complex, non-linear relationships — such as how prolonged high humidity combined with moderate heat degrades transformer lifespans over time.

**Our Solution:** An XGBoost-based ML model that empirically learns multi-variable interactions between weather conditions and physical grid failures, enhanced with infrastructure fragility scoring unique to each city.

**Target Cities:** Lucknow, Noida, Ghaziabad, Agra, Meerut, Firozabad

---

## 🧠 Approach — Cross-Continental Transfer Learning

Since no publicly available Indian outage dataset exists, we leverage a key insight: **the physics of grid failure is universal** — transformers overheat under thermal stress and transmission lines snap under wind loads regardless of geography.

```
US Weather + Real US Outage Data  →  XGBoost Training  →  Trained Model
Indian Weather Data               →  Trained Model     →  Indian Risk Scores
                                                        ↓
                                              DISCOM Fragility Multiplier
                                                        ↓
                                              Adjusted Risk Predictions
```

### Data Sources
| Source | Description | Size |
|---|---|---|
| [EAGLE-I Dataset](https://eagle-i.doe.gov/) (US DOE/ORNL) | Real county-level outage events at 15-min intervals (2023) | 26M rows, 1.2 GB |
| [Open-Meteo Archive API](https://open-meteo.com/) | Hourly historical weather (ERA5 reanalysis) for any global coordinate | ~1.7M matched rows |
| UPERC/PFC Reports (FY 2023-24) | Official DISCOM performance data for UP utilities | 3 DISCOMs |

---

## 🏗️ Pipeline Architecture

### Stage 1 — Data Preparation
| Script | Purpose |
|---|---|
| `src/filter_eagle_i.py` | Filters EAGLE-I to 6 US states with climates matching UP/NCR (TX, AZ, LA, FL, MS, OK) |
| `src/fetch_us_weather.py` | Fetches hourly weather for 20 US cities, merges with outage data via haversine matching |
| `src/engineer_us_features.py` | Engineers 13 features (v1): heat index, temporal flags, seasonal indicators |

### Stage 2 — Model Training & Enhancement
| Script | Purpose |
|---|---|
| `src/train_model.py` | Trains XGBoost v1 (13 features) with cost-sensitive learning (`scale_pos_weight=6.37`) |
| `src/enhance_and_retrain.py` | Adds 13 new features (wind gusts, rolling temps, interaction terms) → v2 (26 features) |
| `src/mrmr_selection.py` | MRMR feature selection — validates all features contribute meaningfully |

### Stage 3 — Inference
| Script | Purpose |
|---|---|
| `src/fetch_weather.py` | Fetches 2 years of hourly weather for 6 UP/NCR cities |
| `src/feature_engineering.py` | Engineers features with Indian seasonal definitions (summer = Apr–Jun) |
| `src/inference.py` | v1 inference on Indian data |
| `src/inference_v2.py` | v2 inference with fragility multipliers and risk level classification |

---

## 📊 Model Performance

### v2 Model (26 features, threshold = 0.55)

| Metric | Value |
|---|---|
| **Accuracy** | **74.4%** |
| **Recall** | **51.6%** |
| **Precision** | **27.0%** |
| **F1 Score** | **0.354** |

> **Design choice:** The model is intentionally tuned for higher recall (catching outages) at the cost of precision (more false alarms). For a safety/alerting tool, missing a real outage is far worse than a false warning.

### Top 5 Features (by importance)
1. `temp_x_humidity` — Combined thermal-moisture stress
2. `is_summer` — Summer = highest risk season
3. `month` — Seasonal patterns
4. `surface_pressure` — Low pressure signals storms
5. `is_monsoon` — Monsoon period flag

### Risk Level Definitions
| Level | Probability | Meaning |
|---|---|---|
| 🟢 LOW | < 30% | Grid is safe |
| 🟡 MODERATE | 30–50% | Elevated caution |
| 🟠 HIGH | 50–70% | Significant outage risk |
| 🔴 CRITICAL | ≥ 70% | Grid failure highly likely |

---

## 🏙️ Infrastructure Fragility Scores

Inspired by [Wang et al. (2024)](https://doi.org/10.xxxx) — same weather causes different outage rates depending on local infrastructure quality. Scores are derived from **official DISCOM distribution loss data** (UPERC/PFC FY 2023-24).

| City | DISCOM | Rating | Fragility Score | Impact on 45% raw risk |
|---|---|---|---|---|
| Noida | PVVNL | A+ | 0.93 | → 41.9% |
| Ghaziabad | PVVNL | A+ | 1.00 | → 45.0% |
| Meerut | PVVNL | A+ | 1.07 | → 48.2% |
| Lucknow | MVVNL | B- | 1.13 | → 50.9% |
| Agra | DVVNL | B- | 1.27 | → 57.2% |
| Firozabad | DVVNL | B- | 1.40 | → **63.0%** |

---

## 📖 Research Foundation

This project is informed by systematic review of 4 academic papers:

1. **Ghasemkhani et al. (2024)** — *Primary blueprint.* XGBoost + MRMR feature selection → 97.66% accuracy on US outage data.
2. **Wang et al. (2024, Wayne State)** — Infrastructure density & socio-economic conditioning → inspired our fragility multiplier.
3. **LSTM-based temporal models** — Sequential outage prediction (cited as future scope).
4. **Graph Neural Networks** — Spatial cascade modeling across substations (cited as future scope).

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- ~2 GB disk space for data files (shared separately)

### Clone & Install
```bash
git clone https://github.com/amisha-srivastavaa/Project_Beyond-the-blackbox.git
cd Project_Beyond-the-blackbox
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data Setup
The `data/` directory is not included in the repository due to file size limits (~2 GB). **Obtain the data files from the team's shared drive** and place them in a `data/` folder at the project root:

```
data/
├── outage_data_2023.csv           # Raw EAGLE-I dataset (1.2 GB)
├── eagle_i_filtered.csv           # Filtered to 6 US states (77 MB)
├── us_training_data.csv           # Outage + weather merged (135 MB)
├── us_training_final.csv          # v1 engineered features (186 MB)
├── us_training_v2.csv             # v2 engineered features (354 MB)
├── weather_data.csv               # UP/NCR raw weather (5 MB)
├── engineered_data.csv            # UP/NCR engineered features (7.7 MB)
├── up_predictions.csv             # v1 predictions (9 MB)
├── up_predictions_v2.csv          # v2 predictions (22 MB)
├── dvvnl_power_map.xlsx           # DVVNL utility reference
└── dvvnl_roaster.pdf              # DVVNL utility reference
```

---

## 📁 Project Structure

```
Project_Beyond-the-blackbox/
│
├── src/                            # All source code
│   ├── filter_eagle_i.py           # Stage 1: Filter EAGLE-I data
│   ├── fetch_us_weather.py         # Stage 1: Fetch & merge US weather
│   ├── engineer_us_features.py     # Stage 1: Feature engineering (US)
│   ├── train_model.py              # Stage 2: Train XGBoost v1
│   ├── enhance_and_retrain.py      # Stage 2: v2 features + retrain
│   ├── mrmr_selection.py           # Stage 2: MRMR feature selection
│   ├── fetch_weather.py            # Stage 3: Fetch UP/NCR weather
│   ├── feature_engineering.py      # Stage 3: Feature engineering (India)
│   ├── inference.py                # Stage 3: v1 inference
│   └── inference_v2.py             # Stage 3: v2 inference
│
├── models/                         # Trained models & artifacts
│   ├── xgboost_model.json          # v1 trained model
│   ├── xgboost_model_v2.json       # v2 trained model (production)
│   ├── training_metrics.json       # v1 evaluation metrics
│   ├── training_metrics_v2.json    # v2 evaluation metrics
│   ├── model_config.json           # Threshold config (0.55)
│   ├── fragility_scores.json       # DISCOM-based city fragility scores
│   ├── mrmr_features.json          # MRMR feature ranking
│   ├── feature_importance.png      # v1 feature importance chart
│   └── threshold_analysis.png      # Threshold optimization chart
│
├── Research/                       # Academic papers & documentation
│   ├── project_progress.md         # Detailed project documentation
│   ├── StrategyChange.md           # Data strategy pivot notes
│   ├── Paper1-4.pdf                # Reference papers
│   └── *.txt                       # Extracted paper text
│
├── data/                           # ⚠️ Not in repo — see Data Setup
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚠️ Known Limitations

1. **Cross-Continental Transfer Gap** — Model learned US grid failure patterns (heat-dominated). Indian grids fail differently (rain, overloaded transformers, poor maintenance).
2. **Weather-Only Features** — No access to utility-specific data (equipment age, maintenance logs, load curves).
3. **Precision Trade-off** — High recall tuning means many warnings won't result in actual outages (~73% false alarm rate).
4. **Class Imbalance** — Only 13.6% of training data represents outage events.

---

## 🚧 Roadmap

| Task | Status |
|---|---|
| Data pipeline (filter → merge → engineer) | ✅ Complete |
| XGBoost v1 training (13 features) | ✅ Complete |
| XGBoost v2 training (26 features) | ✅ Complete |
| MRMR feature selection analysis | ✅ Complete |
| Threshold optimization | ✅ Complete |
| Infrastructure fragility scoring | ✅ Complete |
| Indian city inference (6 cities × 2 years) | ✅ Complete |
| Rain hazard adjustment for Indian conditions | 🔲 Pending |
| Backend API (FastAPI) | 🔲 Pending |
| Frontend Dashboard (risk map + charts) | 🔲 Pending |
| Real-time forecast mode (next 24–48 hours) | 🔲 Pending |
| Push notification system for CRITICAL alerts | 🔲 Pending |

---

## 👥 Team

 Team Simplicity Paradox

---

*For detailed technical documentation, see [`Research/project_progress.md`](Research/project_progress.md).*
