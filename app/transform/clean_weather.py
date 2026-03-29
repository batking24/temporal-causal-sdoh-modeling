"""
clean_weather.py — Clean raw weather data and produce weather_daily_agg.

Cleaning pipeline:
    1. Load raw_weather_daily from SQLite
    2. Handle NULLs via forward-fill + linear interpolation
    3. Cap outliers using IQR-based fencing
    4. Derive extreme-event flags:
        - heatwave_flag: 3+ consecutive days with tmax > 95th percentile
        - coldwave_flag: 3+ consecutive days with tmin < 5th percentile
        - heavy_rain_flag: daily precip > 95th percentile
    5. Compute rolling statistics (7d, 14d, 30d)
    6. Write to weather_daily_agg
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import get_raw_connection

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_raw_weather() -> pd.DataFrame:
    """Load raw_weather_daily into a DataFrame."""
    conn = get_raw_connection()
    df = pd.read_sql(
        "SELECT date, region_id, tmax, tmin, tavg, prcp, snow, awnd "
        "FROM raw_weather_daily ORDER BY region_id, date",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d raw weather rows", len(df))
    return df


def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then interpolate NULLs within each region."""
    numeric_cols = ["tmax", "tmin", "tavg", "prcp", "snow", "awnd"]
    df[numeric_cols] = df.groupby("region_id")[numeric_cols].transform(
        lambda g: g.ffill().bfill().interpolate(method="linear", limit_direction="both")
    )
    nulls_remaining = df[numeric_cols].isnull().sum().sum()
    if nulls_remaining > 0:
        logger.warning("Still %d NULLs after interpolation — filling with 0", nulls_remaining)
        df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers using IQR fencing per region for temperature columns."""
    for col in ["tmax", "tmin", "tavg"]:
        q1 = df.groupby("region_id")[col].transform(lambda x: x.quantile(0.01))
        q99 = df.groupby("region_id")[col].transform(lambda x: x.quantile(0.99))
        before = ((df[col] < q1) | (df[col] > q99)).sum()
        df[col] = df[col].clip(lower=q1, upper=q99)
        if before > 0:
            logger.info("Capped %d outliers in %s", before, col)
    return df


def _derive_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive extreme weather event flags.

    - heatwave_flag: 3+ consecutive days with tmax above 95th percentile
    - coldwave_flag: 3+ consecutive days with tmin below 5th percentile
    - heavy_rain_flag: daily precipitation above 95th percentile
    """
    # Compute per-region percentiles
    tmax_95 = df.groupby("region_id")["tmax"].transform(lambda x: x.quantile(0.95))
    tmin_05 = df.groupby("region_id")["tmin"].transform(lambda x: x.quantile(0.05))
    prcp_95 = df.groupby("region_id")["prcp"].transform(
        lambda x: x[x > 0].quantile(0.95) if (x > 0).any() else 999
    )

    # Daily threshold exceedance
    hot_day = (df["tmax"] >= tmax_95).astype(int)
    cold_day = (df["tmin"] <= tmin_05).astype(int)

    # Consecutive-day detection via rolling sum (3-day window)
    def _consecutive_flag(series: pd.Series, window: int = 3) -> pd.Series:
        """Flag True when the rolling sum of a binary series equals window size."""
        rolling = series.groupby(df["region_id"]).rolling(window, min_periods=window).sum()
        rolling = rolling.reset_index(level=0, drop=True)
        return (rolling >= window).astype(int)

    df["heatwave_flag"] = _consecutive_flag(hot_day)
    df["coldwave_flag"] = _consecutive_flag(cold_day)
    df["heavy_rain_flag"] = (df["prcp"] >= prcp_95).astype(int)

    hw_days = df["heatwave_flag"].sum()
    cw_days = df["coldwave_flag"].sum()
    hr_days = df["heavy_rain_flag"].sum()
    logger.info("Event flags: %d heatwave days, %d coldwave days, %d heavy rain days",
                hw_days, cw_days, hr_days)
    return df


def _compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling temperature and precipitation statistics."""
    df = df.sort_values(["region_id", "date"])

    grouped = df.groupby("region_id")

    df["rolling_7d_temp"] = grouped["tavg"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).round(2)

    df["rolling_14d_temp"] = grouped["tavg"].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    ).round(2)

    df["rolling_7d_precip"] = grouped["prcp"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    ).round(2)

    df["rolling_30d_precip"] = grouped["prcp"].transform(
        lambda x: x.rolling(30, min_periods=1).sum()
    ).round(2)

    return df


def clean_weather() -> int:
    """
    Full weather cleaning pipeline.

    Returns:
        Number of rows written to weather_daily_agg.
    """
    start = time.time()

    df = _load_raw_weather()
    df = _handle_nulls(df)
    df = _cap_outliers(df)
    df = _derive_event_flags(df)
    df = _compute_rolling_stats(df)

    # Compute derived columns
    df["avg_temp"] = df["tavg"].round(2)
    df["max_temp"] = df["tmax"].round(2)
    df["min_temp"] = df["tmin"].round(2)
    df["temp_range"] = (df["tmax"] - df["tmin"]).round(2)
    df["precip"] = df["prcp"].round(2)
    df["wind_speed"] = df["awnd"].round(2)

    # Select final columns
    output_cols = [
        "date", "region_id", "avg_temp", "max_temp", "min_temp", "temp_range",
        "precip", "snow", "wind_speed",
        "heatwave_flag", "coldwave_flag", "heavy_rain_flag",
        "rolling_7d_temp", "rolling_14d_temp", "rolling_7d_precip", "rolling_30d_precip",
    ]
    out = df[output_cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    # Write to DB
    conn = get_raw_connection()
    conn.execute("DELETE FROM weather_daily_agg;")
    out.to_sql("weather_daily_agg", conn, if_exists="append", index=False)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM weather_daily_agg").fetchone()[0]
    conn.close()

    elapsed = round(time.time() - start, 2)
    logger.info("weather_daily_agg: %d rows written in %.2fs", count, elapsed)
    return count


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    count = clean_weather()
    print(f"\n✅ weather_daily_agg populated: {count:,} rows")

    conn = get_raw_connection()
    # Show event flag stats
    stats = conn.execute("""
        SELECT
            SUM(heatwave_flag) as heatwave_days,
            SUM(coldwave_flag) as coldwave_days,
            SUM(heavy_rain_flag) as heavy_rain_days,
            ROUND(AVG(avg_temp), 1) as mean_temp,
            ROUND(AVG(precip), 3) as mean_precip
        FROM weather_daily_agg
    """).fetchone()
    print(f"\n  Heatwave days:    {stats[0]}")
    print(f"  Coldwave days:    {stats[1]}")
    print(f"  Heavy rain days:  {stats[2]}")
    print(f"  Mean temperature: {stats[3]}°F")
    print(f"  Mean daily precip: {stats[4]}\"")

    # Sample
    print("\n  Sample (first 3 rows for CA):")
    for row in conn.execute(
        "SELECT date, avg_temp, max_temp, min_temp, precip, heatwave_flag, rolling_7d_temp "
        "FROM weather_daily_agg WHERE region_id='CA' ORDER BY date LIMIT 3"
    ):
        print(f"    {row}")
    conn.close()
