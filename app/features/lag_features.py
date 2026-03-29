"""
lag_features.py — Generate lagged weather and target features.

For each (region_id, need_type) time series, creates:
    Weather lags:
        tmax_lag_{1,3,7,14,30}
        prcp_lag_{1,3,7,14,30}
    Target lags:
        target_lag_1, target_lag_7

Uses pandas groupby + shift for efficient vectorized computation.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

WEATHER_LAG_DAYS = [1, 3, 7, 14, 30]
TARGET_LAG_DAYS = [1, 7]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged weather and target features to the feature DataFrame.

    Args:
        df: DataFrame with columns: date, region_id, need_type, max_temp, precip, target_count
            Must be sorted by (region_id, need_type, date).

    Returns:
        DataFrame with new lag columns added.
    """
    df = df.sort_values(["region_id", "need_type", "date"]).copy()
    group = df.groupby(["region_id", "need_type"])

    # Weather lags
    for lag in WEATHER_LAG_DAYS:
        df[f"tmax_lag_{lag}"] = group["max_temp"].shift(lag)
        df[f"prcp_lag_{lag}"] = group["precip"].shift(lag)
        logger.debug("Added weather lag %d", lag)

    # Target lags
    for lag in TARGET_LAG_DAYS:
        df[f"target_lag_{lag}"] = group["target_count"].shift(lag)
        logger.debug("Added target lag %d", lag)

    total_lag_cols = len(WEATHER_LAG_DAYS) * 2 + len(TARGET_LAG_DAYS)
    logger.info("Added %d lag feature columns", total_lag_cols)
    return df
