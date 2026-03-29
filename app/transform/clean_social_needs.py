"""
clean_social_needs.py — Clean raw social-needs data and produce social_needs_daily_agg.

Cleaning pipeline:
    1. Load raw_social_needs from SQLite
    2. Parse and normalize dates → date-only (YYYY-MM-DD)
    3. Standardize category names (trim, case-normalize)
    4. Remove duplicates on (ref_id, need_id, source_file)
    5. Map ZIP → state for region_id (state-level aggregation)
    6. Aggregate to daily (date, region_id, need_type) grain:
        - daily_need_count
        - confirmed_count
        - unmet_count
    7. Compute rolling windows: 7d, 14d, 30d
    8. Write to social_needs_daily_agg
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from app.config import get_settings
from app.db import get_raw_connection

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_raw_social_needs() -> pd.DataFrame:
    """Load raw_social_needs into a DataFrame."""
    conn = get_raw_connection()
    df = pd.read_sql(
        "SELECT ref_id, ref_code, ref_date, zipcode, state, category, "
        "need_id, need_status, need_source, source_file "
        "FROM raw_social_needs",
        conn,
    )
    conn.close()
    logger.info("Loaded %d raw social-needs rows", len(df))
    return df


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date-only from datetime strings."""
    df["date"] = pd.to_datetime(df["ref_date"], errors="coerce").dt.date
    invalid = df["date"].isna().sum()
    if invalid > 0:
        logger.warning("Dropped %d rows with unparseable dates", invalid)
        df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize category names."""
    df["category"] = df["category"].str.strip()

    # Standardization map for known variations
    _CATEGORY_MAP = {
        "Other": "Other Assistance",
    }
    df["category"] = df["category"].replace(_CATEGORY_MAP)

    unique_cats = df["category"].nunique()
    logger.info("Standardized to %d unique categories", unique_cats)
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records based on (ref_id, need_id, source_file)."""
    before = len(df)
    df = df.drop_duplicates(subset=["ref_id", "need_id", "source_file"], keep="first")
    after = len(df)
    removed = before - after
    if removed > 0:
        logger.info("Removed %d duplicate rows (%d → %d)", removed, before, after)
    return df


def _assign_region_id(df: pd.DataFrame) -> pd.DataFrame:
    """Set region_id = state for state-level aggregation."""
    df["region_id"] = df["state"]
    missing = df["region_id"].isna().sum()
    if missing > 0:
        logger.warning("Dropping %d rows with no state", missing)
        df = df.dropna(subset=["region_id"])
    return df


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to daily (date, region_id, need_type) → counts.

    Computes:
        - daily_need_count: total needs that day
        - confirmed_count: needs with status='Confirmed'
        - unmet_count: needs with status='Unmet'
    """
    # Compute status flags before aggregation
    df["is_confirmed"] = (df["need_status"] == "Confirmed").astype(int)
    df["is_unmet"] = (df["need_status"] == "Unmet").astype(int)

    agg = df.groupby(["date", "region_id", "category"]).agg(
        daily_need_count=("ref_id", "count"),
        confirmed_count=("is_confirmed", "sum"),
        unmet_count=("is_unmet", "sum"),
    ).reset_index()

    agg.rename(columns={"category": "need_type"}, inplace=True)
    logger.info("Aggregated to %d daily (region, need_type) rows", len(agg))
    return agg


def _compute_rolling_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 7/14/30-day need count averages per (region, need_type)."""
    df = df.sort_values(["region_id", "need_type", "date"])

    grouped = df.groupby(["region_id", "need_type"])

    df["rolling_7d_count"] = grouped["daily_need_count"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).round(2)

    df["rolling_14d_count"] = grouped["daily_need_count"].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    ).round(2)

    df["rolling_30d_count"] = grouped["daily_need_count"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    ).round(2)

    return df


def clean_social_needs() -> int:
    """
    Full social-needs cleaning pipeline.

    Returns:
        Number of rows written to social_needs_daily_agg.
    """
    start = time.time()

    df = _load_raw_social_needs()
    df = _normalize_dates(df)
    df = _standardize_categories(df)
    df = _deduplicate(df)
    df = _assign_region_id(df)
    agg = _aggregate_daily(df)
    agg = _compute_rolling_counts(agg)

    # Format output
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")

    output_cols = [
        "date", "region_id", "need_type",
        "daily_need_count", "confirmed_count", "unmet_count",
        "rolling_7d_count", "rolling_14d_count", "rolling_30d_count",
    ]
    out = agg[output_cols]

    # Write to DB
    conn = get_raw_connection()
    conn.execute("DELETE FROM social_needs_daily_agg;")
    out.to_sql("social_needs_daily_agg", conn, if_exists="append", index=False)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM social_needs_daily_agg").fetchone()[0]
    conn.close()

    elapsed = round(time.time() - start, 2)
    logger.info("social_needs_daily_agg: %d rows written in %.2fs", count, elapsed)
    return count


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    count = clean_social_needs()
    print(f"\n✅ social_needs_daily_agg populated: {count:,} rows")

    conn = get_raw_connection()
    # Stats
    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT region_id) as regions,
            COUNT(DISTINCT need_type) as need_types,
            MIN(date) as min_date,
            MAX(date) as max_date,
            ROUND(AVG(daily_need_count), 1) as avg_daily_count
        FROM social_needs_daily_agg
    """).fetchone()
    print(f"\n  Regions:       {stats[0]}")
    print(f"  Need types:    {stats[1]}")
    print(f"  Date range:    {stats[2]} → {stats[3]}")
    print(f"  Avg daily count: {stats[4]}")

    # Top need types
    print("\n  Daily rows per need type:")
    for row in conn.execute(
        "SELECT need_type, COUNT(*) as rows, SUM(daily_need_count) as total_needs "
        "FROM social_needs_daily_agg GROUP BY need_type ORDER BY total_needs DESC"
    ):
        print(f"    {row[0]:30s} {row[1]:>6,} rows  |  {row[2]:>8,} total needs")

    conn.close()
