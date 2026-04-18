"""
Phase 2: Train XGBoost Model
================================================================
Trains an XGBoost Classifier on the engineered US training dataset.
Uses cost-sensitive learning (scale_pos_weight) to handle the class imbalance.

Input:  data/us_training_final.csv
Output: models/xgboost_model.json
        models/feature_importance.png
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os
import json

# Configuration
INPUT_FILE = "data/us_training_final.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
PLOT_PATH = os.path.join(MODEL_DIR, "feature_importance.png")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")

# Features to use for training
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
    "heat_index"
]
TARGET = "outage"

def main():
    print("=" * 60)
    print("Phase 2: Training XGBoost Model")
    print("=" * 60)

    # 1. Setup Data Directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Load Data
    print(f"Loading dataset: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Checking for missing values in features...")
    # Drop rows with NaN in any of the feature columns
    df = df.dropna(subset=FEATURES + [TARGET])
    
    print(f"Total rows for training: {len(df):,}")

    # 3. Define X and y
    X = df[FEATURES]
    y = df[TARGET]

    # 4. Train/Test Split
    print("Splitting data into 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    outage_count = y_train.sum()
    no_outage_count = len(y_train) - outage_count
    print(f"  Training Set: {len(y_train):,} total ({outage_count:,} outages, {no_outage_count:,} normal)")

    # 5. Handle Imbalance (scale_pos_weight)
    # The formula is typically: number of negative samples / number of positive samples
    scale_pos = no_outage_count / outage_count
    print(f"\nHandling class imbalance...")
    print(f"  Calculated scale_pos_weight: {scale_pos:.2f} (Penalizes missed outages ~{int(scale_pos)}x more)")

    # 6. Initialize Model
    print("\nInitializing XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=200,          # Number of decision trees
        max_depth=6,               # Max depth of each tree (complexity)
        learning_rate=0.1,         # How fast the model learns
        scale_pos_weight=scale_pos,# Cost-sensitive learning for imbalance
        random_state=42,           # For reproducibility
        n_jobs=-1,                 # Use all CPU cores
        eval_metric="logloss"      # Evaluation metric during training
    )

    # 7. Train Model
    print("\nTraining the model (this may take a minute depending on your CPU)...")
    model.fit(X_train, y_train)
    print("✅ Training complete.")

    # 8. Evaluate Model
    print("\nEvaluating model on the 20% test set...")
    y_pred = model.predict(X_test)
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    
    print("=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                     [Predicted Normal]  [Predicted Outage]")
    print(f"[Actual Normal]      {cm[0][0]:>15,}   {cm[0][1]:>15,}")
    print(f"[Actual Outage]      {cm[1][0]:>15,}   {cm[1][1]:>15,}")

    accuracy = accuracy_score(y_test, y_pred)
    
    # 9. Save Model and Metrics
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)

    # Extract metrics for easy reading
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_data = {
        "accuracy": accuracy,
        "outage_class_recall": report["1"]["recall"],
        "outage_class_precision": report["1"]["precision"],
        "outage_class_f1": report["1"]["f1-score"],
        "normal_class_recall": report["0"]["recall"],
        "normal_class_precision": report["0"]["precision"],
        "scale_pos_weight_used": scale_pos
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Metrics saved to {METRICS_PATH}")

    # 10. Feature Importance Plot
    print("\nGenerating feature importance plot...")
    plt.figure(figsize=(10, 8))
    
    # Get feature importances 
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURES[i] for i in indices]
    
    plt.title("XGBoost Feature Importances (Weather -> Grid Outage)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), sorted_features, rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"Saved feature importance plot to {PLOT_PATH}")
    
    print("\n" + "=" * 60)
    print("✅ PHASE 2 COMPLETE: Model is trained and deployed to models/ folder.")
    print("=" * 60)

if __name__ == "__main__":
    main()
