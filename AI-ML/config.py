"""Project-wide configuration constants and paths."""

from pathlib import Path
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
ROC_PLOT_PATH = ARTIFACTS_DIR / "roc_curves.png"
CM_PLOT_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
FI_PLOT_PATH = ARTIFACTS_DIR / "feature_importance.png"

MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "SVM",
    "GradientBoosting",
]

DATA = load_breast_cancer()
FEATURE_NAMES = list(DATA.feature_names)
TARGET_NAMES = list(DATA.target_names)

MEAN_FEATURES = [f for f in FEATURE_NAMES if f.endswith("mean")]
SE_FEATURES = [f for f in FEATURE_NAMES if f.endswith("se")]
WORST_FEATURES = [f for f in FEATURE_NAMES if f.endswith("worst")]
