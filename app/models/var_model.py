"""
var_model.py — Vector Autoregression (VAR) and weather-augmented models.

Two model variants:
    1. var_all_weather: Uses ALL weather features + target lags
    2. var_granger_selected: Uses only Granger-significant weather lags

Both use Ridge regression for stability with moderate-sized datasets.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from app.config import get_settings
from app.db import get_raw_connection

logger = logging.getLogger(__name__)
settings = get_settings()

# All weather feature columns available in model_features_daily
ALL_WEATHER_FEATURES = [
    "tmax_lag_1", "tmax_lag_3", "tmax_lag_7", "tmax_lag_14", "tmax_lag_30",
    "prcp_lag_1", "prcp_lag_3", "prcp_lag_7", "prcp_lag_14", "prcp_lag_30",
    "temp_rollmean_7", "temp_rollmean_14",
    "precip_rollsum_7", "precip_rollsum_30",
    "temp_precip_interact", "temp_target_interact", "temp_trend_7d",
    "heatwave_flag", "coldwave_flag", "heavy_rain_flag",
]

# Target / calendar features (shared across all models)
BASE_FEATURES = [
    "target_lag_1", "target_lag_7",
    "target_rollmean_7", "target_rollmean_14",
    "day_of_week", "month", "is_weekend",
]


def _select_granger_features(
    significant_lags: dict[str, list[int]],
) -> list[str]:
    """
    Map Granger-significant lags back to feature column names.

    Args:
        significant_lags: Dict from granger.get_significant_lags()
            e.g. {"max_temp": [1, 3, 7], "precip": [7, 14]}

    Returns:
        List of feature column names to use.
    """
    feature_map = {
        "max_temp": {1: "tmax_lag_1", 3: "tmax_lag_3", 7: "tmax_lag_7",
                     14: "tmax_lag_14", 30: "tmax_lag_30"},
        "precip": {1: "prcp_lag_1", 3: "prcp_lag_3", 7: "prcp_lag_7",
                   14: "prcp_lag_14", 30: "prcp_lag_30"},
    }

    selected = []
    for var_name, lags in significant_lags.items():
        mapping = feature_map.get(var_name, {})
        for lag in lags:
            # Find closest available lag
            if lag in mapping:
                selected.append(mapping[lag])
            else:
                closest = min(mapping.keys(), key=lambda x: abs(x - lag), default=None)
                if closest is not None:
                    selected.append(mapping[closest])

    # Always include rolling features, event flags, interaction terms, and trends
    selected.extend([
        "temp_rollmean_7", "precip_rollsum_7",
        "heatwave_flag", "coldwave_flag", "heavy_rain_flag",
        "temp_precip_interact", "temp_target_interact", "temp_trend_7d"
    ])

    return list(set(selected))


def train_weather_model(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    weather_features: list[str] | None = None,
    model_name: str = "var_all_weather",
) -> dict:
    """
    Train a weather-augmented prediction model.

    Args:
        df: Full feature DataFrame.
        train_mask: Boolean mask for training rows.
        test_mask: Boolean mask for test rows.
        weather_features: Weather feature columns to use.
            If None, uses ALL_WEATHER_FEATURES.
        model_name: Name for the model variant.

    Returns:
        Dict with predictions, actuals, metrics, and feature importance.
    """
    if weather_features is None:
        weather_features = ALL_WEATHER_FEATURES

    feature_cols = BASE_FEATURES + weather_features
    available = [c for c in feature_cols if c in df.columns]
    target_col = "target_count"

    train = df[train_mask].dropna(subset=available + [target_col])
    test = df[test_mask].dropna(subset=available + [target_col])

    if len(train) < 10 or len(test) < 3:
        logger.warning("Insufficient data for %s: train=%d, test=%d",
                        model_name, len(train), len(test))
        return {"rmse": np.nan, "mae": np.nan, "predictions": [], "actuals": [],
                "model_type": model_name}

    X_train = train[available].values
    y_train = train[target_col].values
    X_test = test[available].values
    y_test = test[target_col].values

    # Upgrade to XGBoost for better capture of non-linear climate signals
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100)

    # Feature importance for XGBoost is based on gain
    importance = dict(zip(available, [float(f) for f in model.feature_importances_.round(4)]))

    return {
        "model_type": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(available),
        "predictions": y_pred.tolist(),
        "actuals": y_test.tolist(),
        "feature_importance": importance,
        "features_used": available,
    }


def train_granger_selected_model(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    significant_lags: dict[str, list[int]],
) -> dict:
    """
    Train a model using only Granger-significant weather features.

    This is the "improved" model that should outperform both:
        - baseline_ar (no weather)
        - var_all_weather (all weather, possibly noisy)

    Args:
        df: Full feature DataFrame.
        train_mask, test_mask: Boolean masks.
        significant_lags: Output from granger.get_significant_lags().

    Returns:
        Model results dict.
    """
    selected = _select_granger_features(significant_lags)
    if not selected:
        logger.warning("No Granger-significant features — falling back to all weather")
        selected = ALL_WEATHER_FEATURES

    logger.info("Granger-selected features: %s", selected)

    return train_weather_model(
        df, train_mask, test_mask,
        weather_features=selected,
        model_name="granger_selected",
    )
