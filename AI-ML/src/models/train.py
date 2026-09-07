"""Model training, evaluation, and selection pipeline."""

import json
from copy import deepcopy

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

import config


def build_models():
    """Return a dict of name -> unfitted classifier with sensible defaults."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=10000, random_state=config.RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            gamma="scale",
            random_state=config.RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.9,
            random_state=config.RANDOM_STATE,
        ),
    }


def train_all(X_train, y_train, X_test, y_test):
    """Train every model; return dicts of fitted models, preds, and probs."""
    models = build_models()
    fitted_models = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        m = deepcopy(model)
        m.fit(X_train, y_train)
        fitted_models[name] = m
        predictions[name] = m.predict(X_test)
        probabilities[name] = m.predict_proba(X_test)[:, 1]

    return fitted_models, predictions, probabilities


def evaluate_models(y_test, predictions, probabilities, save=True):
    """Compute classification metrics for all models and optionally save JSON.

    Returns metrics dict keyed by model name.
    """
    metrics = {}
    for name in predictions.keys():
        y_pred = predictions[name]
        y_prob = probabilities[name]
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

    if save:
        with open(config.METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def select_and_save_best(models, metrics, scaler):
    """Pick best model by roc_auc, persist model + scaler, return (name, model)."""
    best_name = max(metrics.keys(), key=lambda n: metrics[n]["roc_auc"])
    best_model = models[best_name]
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    return best_name, best_model


def print_metrics_table(metrics):
    """Pretty-print a metrics table to stdout."""
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header = f"{'Model':<20}" + "".join(f"{c:>12}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, m in metrics.items():
        row = f"{name:<20}" + "".join(f"{m[c]:>12.4f}" for c in cols)
        print(row)
    print()
