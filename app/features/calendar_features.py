"""
calendar_features.py — Generate calendar and temporal metadata features.

Creates:
    day_of_week:  0=Monday → 6=Sunday
    month:        1-12
    season:       winter/spring/summer/fall
    holiday_flag: 1 for US federal holidays, 0 otherwise
    is_weekend:   1 for Sat/Sun, 0 otherwise
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# US federal holidays for 2025 (hardcoded to avoid dependency)
_US_HOLIDAYS_2025 = {
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
}

_SEASON_MAP = {
    1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall",
    11: "fall", 12: "winter",
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based temporal features.

    Args:
        df: DataFrame with a datetime 'date' column.

    Returns:
        DataFrame with calendar feature columns added.
    """
    dates = pd.to_datetime(df["date"])

    df["day_of_week"] = dates.dt.dayofweek           # 0=Mon, 6=Sun
    df["month"] = dates.dt.month
    df["season"] = dates.dt.month.map(_SEASON_MAP)
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    # Holiday flag
    date_strings = dates.dt.strftime("%Y-%m-%d")
    df["holiday_flag"] = date_strings.isin(_US_HOLIDAYS_2025).astype(int)

    n_holidays = df["holiday_flag"].sum()
    n_weekends = df["is_weekend"].sum()
    logger.info("Calendar features: %d holiday rows, %d weekend rows", n_holidays, n_weekends)
    return df
