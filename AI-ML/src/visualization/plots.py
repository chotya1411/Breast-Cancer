"""Generate and save evaluation visualizations."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve

import config


def plot_roc_curves(y_test, probabilities, save_path=None):
    """Overlay ROC curves for every model and save PNG."""
    save_path = save_path or config.ROC_PLOT_PATH
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, y_prob), color in zip(probabilities.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=name).plot(
            ax=ax, curve_kwargs={"color": color, "linewidth": 2}
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Chance (AUC=0.5)")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Save a normalized confusion matrix for the best model."""
    save_path = save_path or config.CM_PLOT_PATH
    fig, ax = plt.subplots(figsize=(7, 6))
    display_labels = [
        name.capitalize() for name in config.TARGET_NAMES
    ]
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=display_labels,
        cmap="Blues",
        normalize="true",
        ax=ax,
        values_format=".2%",
    )
    ax.set_title("Normalized Confusion Matrix — Best Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_feature_importance(feature_importance, feature_names, top_k=15, save_path=None):
    """Horizontal bar chart of top-k feature importances."""
    save_path = save_path or config.FI_PLOT_PATH
    order = np.argsort(feature_importance)[::-1][:top_k]
    names = [feature_names[i] for i in order]
    values = feature_importance[order]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color="#4c72b0", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def extract_feature_importance(fitted_models, feature_names):
    """Return a 1D importance array, preferring tree-based models, else abs(coef)."""
    for candidate in ("GradientBoosting", "RandomForest"):
        if candidate in fitted_models and hasattr(fitted_models[candidate], "feature_importances_"):
            return np.asarray(fitted_models[candidate].feature_importances_, dtype=float)

    if "LogisticRegression" in fitted_models:
        coef = np.abs(fitted_models["LogisticRegression"].coef_[0])
        return coef / coef.sum()

    return np.ones(len(feature_names)) / len(feature_names)


def generate_all_plots(y_test, predictions, probabilities, fitted_models, feature_names=None):
    """Create and save all three plots. Returns dict of saved paths."""
    feature_names = feature_names or config.FEATURE_NAMES
    roc_path = plot_roc_curves(y_test, probabilities)
    best_name = max(
        predictions.keys(),
        key=lambda n: roc_auc_score_safe(y_test, probabilities[n]),
    )
    cm_path = plot_confusion_matrix(y_test, predictions[best_name])
    importance = extract_feature_importance(fitted_models, feature_names)
    fi_path = plot_feature_importance(importance, feature_names)
    return {"roc": roc_path, "confusion_matrix": cm_path, "feature_importance": fi_path}


def roc_auc_score_safe(y_true, y_prob):
    """Small helper so the module does not require sklearn.metrics import at top."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_prob)
