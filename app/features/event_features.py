"""
event_features.py — Carry forward extreme-weather event flags from weather_daily_agg.

These flags were already computed in Phase 3 cleaning. This module simply
merges them into the feature matrix from the weather data.

Flags:
    heatwave_flag:    3+ consecutive days with tmax > 95th percentile
    coldwave_flag:    3+ consecutive days with tmin < 5th percentile
    heavy_rain_flag:  daily precip > 95th percentile
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_event_features(df: pd.DataFrame, weather_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Merge event flags from weather_daily_agg into the feature DataFrame.

    If flags already exist (from the join), this is a no-op.
    Otherwise, merges from weather_agg on (date, region_id).

    Args:
        df: Feature DataFrame.
        weather_agg: weather_daily_agg DataFrame with event flag columns.

    Returns:
        DataFrame with event flag columns.
    """
    flag_cols = ["heatwave_flag", "coldwave_flag", "heavy_rain_flag"]

    # Check if flags already present
    existing = [c for c in flag_cols if c in df.columns]
    if len(existing) == len(flag_cols):
        logger.info("Event flags already present in feature matrix")
        return df

    # Merge from weather_agg
    missing = [c for c in flag_cols if c not in df.columns]
    merge_cols = ["date", "region_id"] + missing
    weather_flags = weather_agg[merge_cols].drop_duplicates()

    before = len(df)
    df = df.merge(weather_flags, on=["date", "region_id"], how="left")

    # Fill any NaNs with 0 (no event)
    for col in flag_cols:
        df[col] = df[col].fillna(0).astype(int)

    after = len(df)
    logger.info("Event features merged: %d → %d rows, %d flag columns",
                before, after, len(flag_cols))
    return df
