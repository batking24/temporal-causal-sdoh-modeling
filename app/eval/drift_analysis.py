"""
drift_analysis.py — Temporal drift and distribution shift analysis.

Analyzes:
    1. How model error changes across seasons (winter/spring/summer)
    2. Performance during extreme-weather periods vs. normal periods
    3. Relative error increase under distribution shift
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SEASON_MAP = {
    1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall",
    11: "fall", 12: "winter",
}


def analyze_seasonal_drift(
    df: pd.DataFrame,
    model_features: list[str],
) -> pd.DataFrame:
    """
    Analyze how prediction error varies by season.

    Uses a simple train-on-one-season, test-on-next approach.

    Returns:
        DataFrame with columns: season, rmse, mae, n_samples
    """
    df = df.copy()
    df["season"] = pd.to_datetime(df["date"]).dt.month.map(_SEASON_MAP)
    target = "target_count"
    available = [c for c in model_features if c in df.columns]

    results = []
    seasons = ["winter", "spring", "summer"]

    for i in range(len(seasons) - 1):
        train_season = seasons[i]
        test_season = seasons[i + 1]

        train = df[df["season"] == train_season].dropna(subset=available + [target])
        test = df[df["season"] == test_season].dropna(subset=available + [target])

        if len(train) < 10 or len(test) < 5:
            continue

        model = Ridge(alpha=1.0)
        model.fit(train[available].values, train[target].values)

        y_pred = np.maximum(model.predict(test[available].values), 0)
        y_true = test[target].values

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        results.append({
            "train_season": train_season,
            "test_season": test_season,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "n_train": len(train),
            "n_test": len(test),
        })

    logger.info("Seasonal drift: %d season transitions analyzed", len(results))
    return pd.DataFrame(results)


def analyze_extreme_weather_performance(
    df: pd.DataFrame,
    model_features: list[str],
) -> dict:
    """
    Compare model performance during extreme vs. normal weather periods.
    Uses XGBoost to match the main modeling pipeline.

    Returns:
        Dict with normal_rmse, extreme_rmse, and relative_increase.
    """
    from xgboost import XGBRegressor
    
    target = "target_count"
    available = [c for c in model_features if c in df.columns]

    if "heatwave_flag" not in df.columns or "coldwave_flag" not in df.columns:
        return {"normal_rmse": None, "extreme_rmse": None, "relative_increase_pct": None}

    # Split data into extreme and normal periods
    extreme_mask = (df["heatwave_flag"] == 1) | (df["coldwave_flag"] == 1) | (df["heavy_rain_flag"] == 1)
    normal_mask = ~extreme_mask

    normal_df = df[normal_mask].dropna(subset=available + [target])
    extreme_df = df[extreme_mask].dropna(subset=available + [target])

    if len(normal_df) < 50 or len(extreme_df) < 10:
        return {"normal_rmse": None, "extreme_rmse": None, "relative_increase_pct": None}

    # Train on normal, predict on both (Temporal Split)
    train_n = int(len(normal_df) * 0.8)
    train = normal_df.iloc[:train_n]
    test_normal = normal_df.iloc[train_n:]
    test_extreme = extreme_df

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(train[available].values, train[target].values)

    pred_normal = np.maximum(model.predict(test_normal[available].values), 0)
    pred_extreme = np.maximum(model.predict(test_extreme[available].values), 0)

    rmse_normal = float(np.sqrt(np.mean((test_normal[target].values - pred_normal) ** 2)))
    rmse_extreme = float(np.sqrt(np.mean((test_extreme[target].values - pred_extreme) ** 2)))

    # Use a small epsilon to avoid division by zero
    relative_increase = (rmse_extreme - rmse_normal) / max(rmse_normal, 1e-6) * 100

    result = {
        "normal_rmse": round(rmse_normal, 4),
        "extreme_rmse": round(rmse_extreme, 4),
        "relative_increase_pct": round(relative_increase, 2),
        "n_normal_test": len(test_normal),
        "n_extreme": len(test_extreme),
    }

    logger.info("Extreme weather drift analysis complete.")
    return result
