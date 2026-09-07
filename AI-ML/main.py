"""End-to-end training pipeline entry point.

Usage:
    python main.py

Outputs will be saved under ./artifacts/:
    best_model.joblib, scaler.joblib, metrics.json,
    roc_curves.png, confusion_matrix.png, feature_importance.png
"""

from __future__ import annotations

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data.dataset import load_data
from src.data.preprocess import preprocess
from src.models.train import (
    evaluate_models,
    print_metrics_table,
    select_and_save_best,
    train_all,
)
from src.visualization.plots import generate_all_plots


def run_pipeline():
    t0 = time.time()
    print("=" * 70)
    print("  Breast Cancer Malignancy Prediction — Training Pipeline")
    print("=" * 70)

    print("\n[1/5] Loading dataset ...")
    X, y = load_data()
    print(f"      Samples: {X.shape[0]}   Features: {X.shape[1]}")
    print(f"      Classes: {dict(zip(config.TARGET_NAMES, [int((y==i).sum()) for i in range(len(config.TARGET_NAMES))]))}")

    print("\n[2/5] Splitting & scaling data ...")
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
    print(f"      Train: {X_train.shape}   Test: {X_test.shape}")

    print("\n[3/5] Training models: LogisticRegression, RandomForest, SVM, GradientBoosting ...")
    models, predictions, probabilities = train_all(X_train, y_train, X_test, y_test)
    print(f"      Trained {len(models)} models.")

    print("\n[4/5] Evaluating models ...")
    metrics = evaluate_models(y_test, predictions, probabilities, save=True)
    print_metrics_table(metrics)
    print(f"      Metrics saved to -> {config.METRICS_PATH}")

    best_name, best_model = select_and_save_best(models, metrics, scaler)
    print(f"      Best model (by ROC AUC): {best_name}  (acc={metrics[best_name]['accuracy']:.4f})")
    print(f"      Model saved  -> {config.BEST_MODEL_PATH}")
    print(f"      Scaler saved -> {config.SCALER_PATH}")

    print("\n[5/5] Generating visualizations ...")
    paths = generate_all_plots(y_test, predictions, probabilities, models)
    for k, p in paths.items():
        print(f"      {k:>20} -> {p}")

    elapsed = time.time() - t0
    print(f"\nPipeline completed successfully in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
