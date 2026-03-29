"""
aggregate_daily.py — Orchestrator for the full Phase 3 pipeline.

Runs in order:
    1. clean_weather → weather_daily_agg
    2. clean_social_needs → social_needs_daily_agg
    3. validate_alignment → coverage report
"""

from __future__ import annotations

import logging
import time

from app.config import get_settings
from app.transform.clean_weather import clean_weather
from app.transform.clean_social_needs import clean_social_needs
from app.transform.region_align import validate_alignment

logger = logging.getLogger(__name__)
settings = get_settings()


def run_full_pipeline() -> dict:
    """
    Execute the complete cleaning + aggregation + alignment pipeline.

    Returns:
        Summary dict with row counts and alignment report.
    """
    start = time.time()
    results = {}

    # Step 1: Clean weather
    logger.info("=" * 60)
    logger.info("STEP 1: Cleaning weather data")
    logger.info("=" * 60)
    results["weather_rows"] = clean_weather()

    # Step 2: Clean social needs
    logger.info("=" * 60)
    logger.info("STEP 2: Cleaning social-needs data")
    logger.info("=" * 60)
    results["social_needs_rows"] = clean_social_needs()

    # Step 3: Validate alignment
    logger.info("=" * 60)
    logger.info("STEP 3: Validating geographic alignment")
    logger.info("=" * 60)
    results["alignment"] = validate_alignment()

    results["total_elapsed_sec"] = round(time.time() - start, 2)
    return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    results = run_full_pipeline()

    print("\n" + "=" * 60)
    print("PHASE 3 — CLEANING & AGGREGATION COMPLETE")
    print("=" * 60)
    print(f"  weather_daily_agg:       {results['weather_rows']:>8,} rows")
    print(f"  social_needs_daily_agg:  {results['social_needs_rows']:>8,} rows")
    print(f"  Alignment coverage:      {results['alignment']['coverage_pct']}%")
    print(f"  Total elapsed:           {results['total_elapsed_sec']}s")
    print("\n✅ Phase 3 complete.")
