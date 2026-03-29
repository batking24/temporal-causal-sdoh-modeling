"""
baseline_ar.py — Baseline autoregressive model (target-only, no weather).

This is the null-hypothesis baseline: can we predict social needs counts
using ONLY past target values (no weather signals)?

Used for comparison against the weather-augmented models.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.config import get_settings
from app.db import get_raw_connection

logger = logging.getLogger(__name__)
settings = get_settings()


def train_baseline_ar(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
) -> dict:
    """
    Train a target-only autoregressive baseline using Ridge regression.

    Features used: target_lag_1, target_lag_7, target_rollmean_7, target_rollmean_14,
                   day_of_week, month, is_weekend

    Args:
        df: Full feature DataFrame.
        train_mask: Boolean mask for training rows.
        test_mask: Boolean mask for test rows.

    Returns:
        Dict with predictions, actuals, and metrics.
    """
    feature_cols = [
        "target_lag_7",
        "day_of_week", "month", "is_weekend",
    ]
    target_col = "target_count"

    available = [c for c in feature_cols if c in df.columns]

    train = df[train_mask].dropna(subset=available + [target_col])
    test = df[test_mask].dropna(subset=available + [target_col])

    if len(train) < 10 or len(test) < 3:
        logger.warning("Insufficient data: train=%d, test=%d", len(train), len(test))
        return {"rmse": np.nan, "mae": np.nan, "predictions": [], "actuals": []}

    X_train = train[available].values
    y_train = train[target_col].values
    X_test = test[available].values
    y_test = test[target_col].values

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # target counts can't be negative

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100)

    return {
        "model_type": "baseline_ar",
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "n_train": len(train),
        "n_test": len(test),
        "predictions": y_pred.tolist(),
        "actuals": y_test.tolist(),
        "feature_importance": dict(zip(available, model.coef_.round(4).tolist())),
    }
