"""
build_features.py — Orchestrator: builds the complete model_features_daily table.

Pipeline:
    1. Join weather_daily_agg ⋈ social_needs_daily_agg on (date, region_id)
    2. Add lag features (weather + target)
    3. Add rolling features (means, sums)
    4. Add calendar features (day_of_week, month, season, holiday, weekend)
    5. Carry forward event flags (heatwave, coldwave, heavy_rain)
    6. Drop NaN rows from lagging (first MAX_LAG_DAYS rows per group)
    7. Write to model_features_daily + export to Parquet
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd

from app.config import get_settings
from app.db import get_raw_connection
from app.features.lag_features import add_lag_features
from app.features.rolling_features import add_rolling_features
from app.features.calendar_features import add_calendar_features
from app.features.event_features import add_event_features

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_weather_agg() -> pd.DataFrame:
    """Load weather_daily_agg from DB."""
    conn = get_raw_connection()
    df = pd.read_sql("SELECT * FROM weather_daily_agg", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d weather_daily_agg rows", len(df))
    return df


def _load_social_needs_agg() -> pd.DataFrame:
    """Load social_needs_daily_agg from DB."""
    conn = get_raw_connection()
    df = pd.read_sql("SELECT * FROM social_needs_daily_agg", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d social_needs_daily_agg rows", len(df))
    return df


def _join_datasets(weather: pd.DataFrame, social: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join weather and social needs on (date, region_id).

    Renames columns for the feature matrix:
        daily_need_count → target_count
        max_temp, precip carry forward as-is
    """
    merged = social.merge(
        weather,
        on=["date", "region_id"],
        how="inner",
        suffixes=("", "_wx"),
    )

    merged.rename(columns={"daily_need_count": "target_count"}, inplace=True)

    logger.info("Joined datasets: %d rows (%d social × %d weather → %d matched)",
                len(merged), len(social), len(weather), len(merged))
    return merged


def build_features() -> int:
    """
    Build the complete feature matrix and write to model_features_daily.

    Returns:
        Number of rows written.
    """
    start = time.time()

    # 1. Load aggregated data
    weather = _load_weather_agg()
    social = _load_social_needs_agg()

    # 2. Join
    df = _join_datasets(weather, social)

    # Ensure max_temp and precip columns exist for feature functions
    if "max_temp" not in df.columns and "max_temp_wx" in df.columns:
        df.rename(columns={"max_temp_wx": "max_temp"}, inplace=True)
    if "precip" not in df.columns and "precip_wx" in df.columns:
        df.rename(columns={"precip_wx": "precip"}, inplace=True)

    # 3. Add lag features
    logger.info("Adding lag features...")
    df = add_lag_features(df)

    # 4. Add rolling features
    logger.info("Adding rolling features...")
    df = add_rolling_features(df)

    # 5. Add calendar features
    logger.info("Adding calendar features...")
    df = add_calendar_features(df)

    # 6. Add event features
    logger.info("Adding event features...")
    df = add_event_features(df, weather)

    # 7. Add Interaction & Trend features
    logger.info("Adding interaction and trend features...")
    df["temp_precip_interact"] = df["temp_rollmean_7"] * df["precip_rollsum_7"]
    df["temp_target_interact"] = df["temp_rollmean_7"] * df["target_rollmean_7"]
    if "tmax_lag_7" in df.columns:
        df["temp_trend_7d"] = df["max_temp"] - df["tmax_lag_7"]
    else:
        df["temp_trend_7d"] = 0

    # 8. Drop rows with NaN from lagging (first N rows per group)
    before_drop = len(df)
    lag_cols = [c for c in df.columns if "_lag_" in c]
    df = df.dropna(subset=lag_cols)
    after_drop = len(df)
    logger.info("Dropped %d rows with NaN lags (%d → %d)",
                before_drop - after_drop, before_drop, after_drop)

    # 9. Select final columns for model_features_daily
    final_cols = [
        "date", "region_id", "need_type", "target_count",
        # lagged weather
        "tmax_lag_1", "tmax_lag_3", "tmax_lag_7", "tmax_lag_14", "tmax_lag_30",
        "prcp_lag_1", "prcp_lag_3", "prcp_lag_7", "prcp_lag_14", "prcp_lag_30",
        # rolling weather
        "temp_rollmean_7", "temp_rollmean_14", "precip_rollsum_7", "precip_rollsum_30",
        # interaction & trend
        "temp_precip_interact", "temp_target_interact", "temp_trend_7d",
        # lagged target
        "target_lag_1", "target_lag_7", "target_rollmean_7", "target_rollmean_14",
        # event flags
        "heatwave_flag", "coldwave_flag", "heavy_rain_flag",
        # calendar
        "day_of_week", "month", "season", "holiday_flag", "is_weekend",
    ]

    # Only keep columns that exist
    available = [c for c in final_cols if c in df.columns]
    missing = set(final_cols) - set(available)
    if missing:
        logger.warning("Missing expected columns: %s", missing)

    out = df[available].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    # Convert numeric types
    int_cols = ["target_count", "target_lag_1", "target_lag_7",
                "heatwave_flag", "coldwave_flag", "heavy_rain_flag",
                "day_of_week", "month", "holiday_flag", "is_weekend"]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].astype(int)

    # 9. Write to DB
    conn = get_raw_connection()
    conn.execute("DELETE FROM model_features_daily;")
    out.to_sql("model_features_daily", conn, if_exists="append", index=False)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM model_features_daily").fetchone()[0]
    conn.close()

    # 10. Export to Parquet
    parquet_path = os.path.join(settings.DATA_PROCESSED_DIR, "features.parquet")
    out.to_parquet(parquet_path, index=False)
    logger.info("Exported features to %s", parquet_path)

    elapsed = round(time.time() - start, 2)
    logger.info("model_features_daily: %d rows, %d columns in %.2fs",
                count, len(available), elapsed)
    return count


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    count = build_features()

    print(f"\n✅ model_features_daily populated: {count:,} rows")

    # Summary stats
    conn = get_raw_connection()

    print("\n  Feature matrix shape:")
    cols = conn.execute("PRAGMA table_info(model_features_daily)").fetchall()
    print(f"    {count:,} rows × {len(cols)} columns")

    print("\n  Columns:")
    for c in cols:
        print(f"    {c[1]:25s} ({c[2]})")

    print("\n  Need types in features:")
    for row in conn.execute(
        "SELECT need_type, COUNT(*) as n FROM model_features_daily "
        "GROUP BY need_type ORDER BY n DESC"
    ):
        print(f"    {row[0]:30s} {row[1]:>5,} rows")

    print("\n  Sample row (first):")
    sample = conn.execute(
        "SELECT date, region_id, need_type, target_count, tmax_lag_1, tmax_lag_7, "
        "target_lag_1, heatwave_flag, day_of_week, season "
        "FROM model_features_daily LIMIT 1"
    ).fetchone()
    print(f"    date={sample[0]}, region={sample[1]}, need={sample[2]}")
    print(f"    target={sample[3]}, tmax_lag1={sample[4]}, tmax_lag7={sample[5]}")
    print(f"    target_lag1={sample[6]}, heatwave={sample[7]}, dow={sample[8]}, season={sample[9]}")

    # Feature stats
    print("\n  Feature value ranges:")
    for col in ["tmax_lag_1", "prcp_lag_7", "target_lag_1", "temp_rollmean_7", "precip_rollsum_30"]:
        stats = conn.execute(
            f"SELECT ROUND(MIN({col}),2), ROUND(AVG({col}),2), ROUND(MAX({col}),2) "
            f"FROM model_features_daily"
        ).fetchone()
        print(f"    {col:25s}  min={stats[0]:>8}  avg={stats[1]:>8}  max={stats[2]:>8}")

    conn.close()
