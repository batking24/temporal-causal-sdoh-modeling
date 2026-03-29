"""
region_align.py — Validate geographic alignment between weather and social-needs data.

Checks:
    1. Every state in social_needs_daily_agg has matching weather data
    2. Every date in social needs has matching weather (coverage report)
    3. Identifies and logs mismatched regions
    4. Produces a coverage report with % of (date, region) pairs matched
"""

from __future__ import annotations

import logging

import pandas as pd

from app.config import get_settings
from app.db import get_raw_connection

logger = logging.getLogger(__name__)
settings = get_settings()


def validate_alignment() -> dict:
    """
    Validate that weather and social-needs data are aligned geographically
    and temporally.

    Returns:
        Alignment report dict with coverage statistics.
    """
    conn = get_raw_connection()

    # 1. State-level check
    wx_states = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT region_id FROM weather_daily_agg"
        ).fetchall()
    )
    sn_states = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT region_id FROM social_needs_daily_agg"
        ).fetchall()
    )

    missing_weather = sn_states - wx_states
    extra_weather = wx_states - sn_states
    matched_states = sn_states & wx_states

    if missing_weather:
        logger.warning("Social-needs states without weather: %s", missing_weather)
    if extra_weather:
        logger.info("Weather states without social needs: %s", extra_weather)

    # 2. Date coverage check
    wx_dates = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM weather_daily_agg"
        ).fetchall()
    )
    sn_dates = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM social_needs_daily_agg"
        ).fetchall()
    )

    date_overlap = sn_dates & wx_dates
    sn_only_dates = sn_dates - wx_dates

    # 3. Full (date, region) pair coverage
    sn_pairs = conn.execute(
        "SELECT COUNT(DISTINCT date || '|' || region_id) FROM social_needs_daily_agg"
    ).fetchone()[0]

    matched_pairs = conn.execute("""
        SELECT COUNT(DISTINCT s.date || '|' || s.region_id)
        FROM social_needs_daily_agg s
        INNER JOIN weather_daily_agg w
            ON s.date = w.date AND s.region_id = w.region_id
    """).fetchone()[0]

    conn.close()

    coverage_pct = round(matched_pairs / max(sn_pairs, 1) * 100, 1)

    report = {
        "social_needs_states": len(sn_states),
        "weather_states": len(wx_states),
        "matched_states": len(matched_states),
        "missing_weather_states": sorted(missing_weather),
        "social_needs_dates": len(sn_dates),
        "weather_dates": len(wx_dates),
        "overlapping_dates": len(date_overlap),
        "social_needs_pairs": sn_pairs,
        "matched_pairs": matched_pairs,
        "coverage_pct": coverage_pct,
    }

    logger.info(
        "Alignment: %d/%d states matched, %d/%d (date,region) pairs covered (%.1f%%)",
        len(matched_states), len(sn_states), matched_pairs, sn_pairs, coverage_pct,
    )

    return report


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    report = validate_alignment()

    print("\n" + "=" * 60)
    print("GEOGRAPHIC & TEMPORAL ALIGNMENT REPORT")
    print("=" * 60)
    print(f"  Social-needs states: {report['social_needs_states']}")
    print(f"  Weather states:      {report['weather_states']}")
    print(f"  Matched states:      {report['matched_states']}")
    if report["missing_weather_states"]:
        print(f"  ⚠️  Missing weather:  {report['missing_weather_states']}")
    print(f"\n  Social-needs dates:  {report['social_needs_dates']}")
    print(f"  Weather dates:       {report['weather_dates']}")
    print(f"  Overlapping dates:   {report['overlapping_dates']}")
    print(f"\n  (date, region) pairs in social needs: {report['social_needs_pairs']:,}")
    print(f"  Matched with weather:                 {report['matched_pairs']:,}")
    print(f"  Coverage:                             {report['coverage_pct']}%")

    if report["coverage_pct"] >= 95:
        print("\n✅ Alignment is excellent.")
    elif report["coverage_pct"] >= 80:
        print("\n⚠️  Alignment is good but has gaps.")
    else:
        print("\n❌ Alignment needs work — significant gaps.")
