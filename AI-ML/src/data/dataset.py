"""Load the Wisconsin Breast Cancer dataset as pandas structures."""

import pandas as pd
from sklearn.datasets import load_breast_cancer

import config


def load_data():
    """Return (X, y) where X is a DataFrame of features and y is a Series of labels.

    Shapes: X (569, 30), y (569,). y=0 -> malignant, y=1 -> benign.
    Feature and target names match config.FEATURE_NAMES and config.TARGET_NAMES.
    """
    bunch = load_breast_cancer()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")
    return X, y


def load_dataframe_with_target():
    """Return a single DataFrame with features and a 'target' column."""
    X, y = load_data()
    df = X.copy()
    df["target"] = y.values
    df["diagnosis"] = df["target"].map(
        {i: name for i, name in enumerate(config.TARGET_NAMES)}
    )
    return df
