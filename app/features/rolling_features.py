"""
rolling_features.py — Generate rolling window statistics.

Creates:
    Weather rolling:
        temp_rollmean_7, temp_rollmean_14
        precip_rollsum_7, precip_rollsum_30
    Target rolling:
        target_rollmean_7, target_rollmean_14
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window features for weather and target series.

    Args:
        df: DataFrame sorted by (region_id, need_type, date), must have
            max_temp, precip, target_count columns.

    Returns:
        DataFrame with new rolling columns.
    """
    df = df.sort_values(["region_id", "need_type", "date"]).copy()
    group = df.groupby(["region_id", "need_type"])

    # Weather rolling means
    df["temp_rollmean_7"] = group["max_temp"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).round(2)

    df["temp_rollmean_14"] = group["max_temp"].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    ).round(2)

    # Precipitation rolling sums
    df["precip_rollsum_7"] = group["precip"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    ).round(2)

    df["precip_rollsum_30"] = group["precip"].transform(
        lambda x: x.rolling(30, min_periods=1).sum()
    ).round(2)

    # Target rolling means
    df["target_rollmean_7"] = group["target_count"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).round(2)

    df["target_rollmean_14"] = group["target_count"].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    ).round(2)

    logger.info("Added 6 rolling feature columns")
    return df
