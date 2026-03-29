"""
region_lookup.py — Build the region_lookup table from ingested social-needs data.

Strategy:
    - Extract unique (ZIPCODE, STATE) pairs from the GroundGame CSVs
    - Use STATE as the primary region_id for modeling (state-level aggregation)
    - Store ZIP → state mapping for geographic alignment
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
from pathlib import Path

from app.config import get_settings
from app.db import get_raw_connection

logger = logging.getLogger(__name__)
settings = get_settings()


def _scan_csv_for_regions(filepath: str) -> list[tuple[str, str, str]]:
    """
    Read a CSV and extract unique (zipcode, state, city) tuples.
    Handles both schema variants in the GroundGame data.
    """
    regions = set()
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zipcode = (row.get("ZIPCODE") or "").strip()
            state = (row.get("STATE") or "").strip()
            if zipcode and state:
                regions.add((zipcode, state))
    return [(z, s, "") for z, s in regions]


def build_region_lookup() -> int:
    """
    Scan all GroundGame CSVs and build a region_lookup table.

    Returns:
        Number of unique regions inserted.
    """
    source_dir = settings.SOURCE_DATA_DIR
    csv_files = [
        "gap_closure_with_program_data.csv",
        "gaps_close_data.csv",
        "gaps_close_met_needs.csv",
        "gaps_close_preproc.csv",
    ]

    all_regions: dict[str, dict] = {}

    for csv_file in csv_files:
        filepath = os.path.join(source_dir, csv_file)
        if not os.path.exists(filepath):
            logger.warning("File not found, skipping: %s", filepath)
            continue

        logger.info("Scanning %s for region data...", csv_file)
        tuples = _scan_csv_for_regions(filepath)

        for zipcode, state, city in tuples:
            key = zipcode
            if key not in all_regions:
                all_regions[key] = {
                    "zipcode": zipcode,
                    "state": state,
                    "city": city,
                    "region_id": state,  # state-level aggregation
                }

    logger.info("Found %d unique ZIP codes across %d states",
                len(all_regions),
                len({r["state"] for r in all_regions.values()}))

    # Bulk insert
    conn = get_raw_connection()
    try:
        conn.execute("DELETE FROM region_lookup;")  # idempotent
        rows = list(all_regions.values())
        for r in rows:
            r["region_id"] = r["zipcode"]   # use ZIP as PK, state for grouping
        conn.executemany(
            """INSERT OR IGNORE INTO region_lookup
               (region_id, zipcode, county_fips, county_name, state, city)
               VALUES (:region_id, :zipcode, NULL, NULL, :state, :city)""",
            rows,
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM region_lookup").fetchone()[0]
        logger.info("Inserted %d rows into region_lookup", count)
        return count
    finally:
        conn.close()


def get_zip_to_state_map() -> dict[str, str]:
    """Return a {zipcode: state} mapping from the region_lookup table."""
    conn = get_raw_connection()
    try:
        rows = conn.execute("SELECT zipcode, state FROM region_lookup").fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    from app.db import init_db
    init_db()
    count = build_region_lookup()
    print(f"\n✅ region_lookup populated: {count} ZIP codes")

    # Print summary
    conn = get_raw_connection()
    states = conn.execute(
        "SELECT state, COUNT(*) as n FROM region_lookup GROUP BY state ORDER BY n DESC"
    ).fetchall()
    conn.close()
    print("\nZIPs per state:")
    for state, n in states:
        print(f"  {state}: {n}")
