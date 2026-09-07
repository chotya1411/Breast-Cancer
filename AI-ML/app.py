"""Streamlit web app: Breast Cancer Malignancy Prediction System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib

import config
from src.data.dataset import load_data, load_dataframe_with_target


st.set_page_config(
    page_title="Breast Cancer Malignancy Predictor",
    page_icon="🔬",
    layout="wide",
)

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    """Load persisted model + scaler from artifacts dir."""
    if not (config.BEST_MODEL_PATH.exists() and config.SCALER_PATH.exists()):
        return None, None
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler


@st.cache_data(show_spinner=False)
def load_metrics():
    if not config.METRICS_PATH.exists():
        return None
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data(), load_dataframe_with_target()


def predict(feature_vector, model, scaler):
    """Predict class and probability for a raw feature vector.

    Returns (class_label: str, probability: float)
    """
    arr = np.asarray(feature_vector, dtype=float).reshape(1, -1)
    scaled = scaler.transform(arr)
    prob = float(model.predict_proba(scaled)[0, 1])
    label_idx = int(model.predict(scaled)[0])
    label = config.TARGET_NAMES[label_idx].capitalize()
    return label, prob


# ---------------------------------------------------------------------------
# Sidebar — page navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dataset Explorer", "📈 Model Comparison", "🔮 Interactive Predictor"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Tip**: First run `python main.py` in the project root "
    "to train models and generate artifacts."
)

if page == "📊 Dataset Explorer":
    st.title("Breast Cancer Dataset Explorer")
    st.caption("Wisconsin Diagnostic Breast Cancer (WDBC) dataset — 569 samples, 30 features")

    (X, y), df = get_dataset()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", f"{X.shape[0]}")
    col2.metric("Features", f"{X.shape[1]}")
    col3.metric("Malignant (0)", f"{int((y == 0).sum())}")
    col4.metric("Benign (1)", f"{int((y == 1).sum())}")

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Class Distribution")
    dist = df["diagnosis"].value_counts().rename_axis("Diagnosis").reset_index(name="Count")
    st.bar_chart(dist.set_index("Diagnosis"))

    st.subheader("Feature Summary Statistics")
    st.dataframe(X.describe().T.round(3), use_container_width=True)


elif page == "📈 Model Comparison":
    st.title("Model Comparison Dashboard")
    metrics = load_metrics()
    if metrics is None:
        st.warning(
            "No metrics found. Please run `python main.py` first to train the models."
        )
        st.stop()

    st.subheader("Evaluation Metrics")
    mdf = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(mdf.style.format("{:.4f}", subset=mdf.columns[1:]), use_container_width=True)

    st.subheader("Accuracy & ROC AUC by Model")
    chart_df = mdf[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    chart_df = chart_df.set_index("Model")
    st.bar_chart(chart_df)

    best = max(metrics.keys(), key=lambda n: metrics[n]["roc_auc"])
    st.success(f"🏆 Best Model: **{best}**  (ROC AUC = {metrics[best]['roc_auc']:.4f})")

    col1, col2 = st.columns(2)
    if config.ROC_PLOT_PATH.exists():
        with col1:
            st.subheader("ROC Curves — All Models")
            st.image(str(config.ROC_PLOT_PATH), use_container_width=True)
    if config.CM_PLOT_PATH.exists():
        with col2:
            st.subheader("Confusion Matrix — Best Model")
            st.image(str(config.CM_PLOT_PATH), use_container_width=True)
    if config.FI_PLOT_PATH.exists():
        st.subheader("Top Feature Importances")
        st.image(str(config.FI_PLOT_PATH), caption="From GradientBoosting / RandomForest feature_importances_ or |LR coefficients|", use_container_width=True)


elif page == "🔮 Interactive Predictor":
    st.title("Interactive Malignancy Predictor")
    st.caption(
        "Adjust the 30 cell nucleus features below and press **Predict** to see "
        "whether the tumor is classified as Benign or Malignant."
    )

    model, scaler = load_model_and_scaler()
    if model is None:
        st.warning(
            "No trained model found. Please run `python main.py` first to train and save the model."
        )
        st.stop()

    (X, _), _ = get_dataset()
    mins = X.min().values
    maxs = X.max().values
    meds = X.median().values

    with st.expander("ℹ️ How are defaults chosen?", expanded=False):
        st.write(
            "Each slider defaults to the **median** value in the dataset, and its "
            "range spans the dataset min to max. You can group the sliders by "
            "Mean / Standard Error / Worst statistics."
        )

    group_labels = {
        "Mean (average values)": config.MEAN_FEATURES,
        "Standard Error": config.SE_FEATURES,
        "Worst (largest values)": config.WORST_FEATURES,
    }

    feature_values = {}
    cols_per_group = 2

    for group_name, group_features in group_labels.items():
        st.markdown(f"#### {group_name}")
        cols = st.columns(cols_per_group)
        for i, feat in enumerate(group_features):
            idx = config.FEATURE_NAMES.index(feat)
            min_v = float(mins[idx])
            max_v = float(maxs[idx])
            med_v = float(meds[idx])
            step = (max_v - min_v) / 500 if max_v > min_v else 0.01
            with cols[i % cols_per_group]:
                feature_values[feat] = st.slider(
                    feat,
                    min_value=round(min_v, 6),
                    max_value=round(max_v, 6),
                    value=round(med_v, 6),
                    step=round(step, 6),
                    key=f"slider_{feat}",
                )

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 3, 1])
    predict_btn = c2.button("🔍 Predict Diagnosis", type="primary", use_container_width=True)

    if predict_btn:
        vec = [feature_values[f] for f in config.FEATURE_NAMES]
        label, prob = predict(vec, model, scaler)

        c2.markdown("## Result")
        if label == "Benign":
            c2.success(f"### ✅ Predicted: **{label}**")
        else:
            c2.error(f"### ⚠️ Predicted: **{label}**")

        pct = prob * 100
        c2.markdown(
            f"**Probability of being Benign**: {pct:.1f}%  "
            f"(probability of Malignant: {100 - pct:.1f}%)"
        )
        c2.progress(float(prob) if label == "Benign" else float(1 - prob))

        with c2.expander("Show input feature values", expanded=False):
            st.json({f: round(v, 5) for f, v in zip(config.FEATURE_NAMES, vec)})


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "🔬 Breast Cancer Malignancy Prediction System — built with "
    "scikit-learn, pandas, matplotlib, and Streamlit."
)
