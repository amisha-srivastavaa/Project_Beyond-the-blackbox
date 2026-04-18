"""
MRMR Feature Selection + Model Retraining
================================================================
Applies MRMR (Minimum Redundancy Maximum Relevance) feature selection
to find the optimal feature subset, then retrains the XGBoost model
and compares with the original.

Based on Paper 1: Ghasemkhani et al. (2024)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mrmr import mrmr_classif
import json

# ── 1. Load data ──

print("=" * 60)
print("MRMR Feature Selection + Model Retraining")
print("=" * 60)

df = pd.read_csv("data/us_training_final.csv").dropna()

ALL_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m",
    "cloud_cover", "surface_pressure", "hour_of_day", "day_of_week", "month",
    "is_summer", "is_monsoon", "is_peak_hour", "heat_index"
]
TARGET = "outage"

X = df[ALL_FEATURES]
y = df[TARGET]

# ── 2. Run MRMR ──

print(f"\nRunning MRMR feature selection...")
print(f"  Input: {len(ALL_FEATURES)} features")

# Rank ALL features using MRMR
selected_features = mrmr_classif(X=X, y=y, K=len(ALL_FEATURES), return_scores=False)

print(f"\n  MRMR Feature Ranking (most important to least):")
for i, feat in enumerate(selected_features):
    print(f"    {i+1:>2}. {feat}")

# ── 3. Train/test split ──

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

no_outage = (y_train == 0).sum()
outage = (y_train == 1).sum()
scale_pos = no_outage / outage

# ── 4. Test different K values ──

print("\n" + "=" * 60)
print("Testing different feature counts (K)")
print("=" * 60)

results = {}
for K in [6, 8, 10, 13]:
    features_k = selected_features[:K]

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    model.fit(X_train[features_k], y_train)

    y_pred = model.predict(X_test[features_k])
    report = classification_report(y_test, y_pred, output_dict=True)

    recall = report["1"]["recall"]
    precision = report["1"]["precision"]
    f1 = report["1"]["f1-score"]
    accuracy = report["accuracy"]

    results[K] = {
        "features": features_k,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "model": model,
    }

    feat_list = ", ".join(features_k)
    print(f"\n  K={K:>2} features: [{feat_list}]")
    print(f"       Recall: {recall:.3f} | Precision: {precision:.3f} | F1: {f1:.3f} | Accuracy: {accuracy:.3f}")

# ── 5. Compare with original ──

print("\n" + "=" * 60)
print("COMPARISON: Original vs MRMR")
print("=" * 60)

with open("models/training_metrics.json") as f:
    old_metrics = json.load(f)

header = f"  {'Model':<20} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Accuracy':>10}"
print(f"\n{header}")
print(f"  {'-' * 56}")

orig_label = "Original (K=13)"
print(f"  {orig_label:<20} {old_metrics['outage_class_recall']:>8.3f} "
      f"{old_metrics['outage_class_precision']:>10.3f} "
      f"{old_metrics['outage_class_f1']:>8.3f} "
      f"{old_metrics['accuracy']:>10.3f}")

best_k = None
best_f1 = old_metrics["outage_class_f1"]
for K, r in results.items():
    label = f"MRMR (K={K})"
    marker = ""
    if r["f1"] > best_f1:
        best_f1 = r["f1"]
        best_k = K
        marker = " <-- BEST"
    print(f"  {label:<20} {r['recall']:>8.3f} {r['precision']:>10.3f} "
          f"{r['f1']:>8.3f} {r['accuracy']:>10.3f}{marker}")

# ── 6. Save best model ──

if best_k:
    print(f"\n✅ MRMR K={best_k} improved F1 score! Saving new model...")
    best = results[best_k]
    best["model"].save_model("models/xgboost_model_mrmr.json")

    with open("models/mrmr_features.json", "w") as f:
        json.dump({"selected_features": list(best["features"]), "K": best_k}, f, indent=4)

    new_metrics = {
        "accuracy": best["accuracy"],
        "outage_class_recall": best["recall"],
        "outage_class_precision": best["precision"],
        "outage_class_f1": best["f1"],
        "scale_pos_weight_used": scale_pos,
        "mrmr_features": list(best["features"]),
        "mrmr_K": best_k,
    }
    with open("models/training_metrics_mrmr.json", "w") as f:
        json.dump(new_metrics, f, indent=4)

    print(f"  Saved: models/xgboost_model_mrmr.json")
    print(f"  Saved: models/mrmr_features.json")
    print(f"  Saved: models/training_metrics_mrmr.json")
else:
    print(f"\n⚠️ MRMR did not improve F1 over the original model.")
    print(f"  Original model remains the best.")

    with open("models/mrmr_features.json", "w") as f:
        json.dump({
            "full_ranking": list(selected_features),
            "note": "MRMR ranking saved for documentation. Original model kept."
        }, f, indent=4)
    print(f"  Saved MRMR ranking to models/mrmr_features.json for documentation")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
