"""Train-test split and standardization pipeline."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def preprocess(X, y):
    """Split data and fit a StandardScaler on the training fold.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler
