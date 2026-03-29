"""
weather_ingest.py — Ingest weather data into raw_weather_daily.

Strategy:
    1. Try NOAA GHCN-D API if NOAA_API_TOKEN is set.
    2. Fall back to realistic synthetic weather generation when API is unavailable.

Synthetic generation produces daily observations per state with:
    - Latitude-based temperature baselines (FL hotter than ME)
    - Realistic seasonal curves (sinusoidal annual cycle)
    - Day-to-day autocorrelation (weather persists)
    - Regional precipitation patterns
    - Proper ranges for each variable
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from datetime import date, datetime, timedelta

import numpy as np

from app.config import get_settings
from app.db import get_raw_connection, init_db

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# State metadata for realistic synthetic generation
# ---------------------------------------------------------------------------
# Approximate latitude, avg annual temp (°F), avg annual precip (inches/day)
_STATE_CLIMATE = {
    "AZ": {"lat": 34.0, "base_temp": 72, "precip_rate": 0.022, "snow_rate": 0.001},
    "CA": {"lat": 36.8, "base_temp": 62, "precip_rate": 0.035, "snow_rate": 0.003},
    "CO": {"lat": 39.0, "base_temp": 50, "precip_rate": 0.040, "snow_rate": 0.030},
    "CT": {"lat": 41.6, "base_temp": 50, "precip_rate": 0.120, "snow_rate": 0.040},
    "DC": {"lat": 38.9, "base_temp": 56, "precip_rate": 0.100, "snow_rate": 0.015},
    "FL": {"lat": 27.8, "base_temp": 72, "precip_rate": 0.130, "snow_rate": 0.000},
    "GA": {"lat": 33.0, "base_temp": 64, "precip_rate": 0.120, "snow_rate": 0.003},
    "IA": {"lat": 42.0, "base_temp": 48, "precip_rate": 0.090, "snow_rate": 0.035},
    "IN": {"lat": 39.8, "base_temp": 52, "precip_rate": 0.100, "snow_rate": 0.025},
    "KY": {"lat": 37.8, "base_temp": 56, "precip_rate": 0.110, "snow_rate": 0.015},
    "LA": {"lat": 31.0, "base_temp": 68, "precip_rate": 0.150, "snow_rate": 0.001},
    "ME": {"lat": 45.0, "base_temp": 42, "precip_rate": 0.100, "snow_rate": 0.060},
    "MO": {"lat": 38.5, "base_temp": 56, "precip_rate": 0.100, "snow_rate": 0.020},
    "NH": {"lat": 43.5, "base_temp": 44, "precip_rate": 0.100, "snow_rate": 0.055},
    "NJ": {"lat": 40.2, "base_temp": 53, "precip_rate": 0.110, "snow_rate": 0.025},
    "NV": {"lat": 39.5, "base_temp": 55, "precip_rate": 0.020, "snow_rate": 0.010},
    "NY": {"lat": 43.0, "base_temp": 48, "precip_rate": 0.100, "snow_rate": 0.040},
    "OH": {"lat": 40.4, "base_temp": 51, "precip_rate": 0.095, "snow_rate": 0.030},
    "TN": {"lat": 35.5, "base_temp": 58, "precip_rate": 0.120, "snow_rate": 0.008},
    "TX": {"lat": 31.0, "base_temp": 66, "precip_rate": 0.080, "snow_rate": 0.002},
    "VA": {"lat": 37.5, "base_temp": 56, "precip_rate": 0.100, "snow_rate": 0.012},
    "WA": {"lat": 47.5, "base_temp": 50, "precip_rate": 0.090, "snow_rate": 0.020},
    "WI": {"lat": 44.5, "base_temp": 44, "precip_rate": 0.080, "snow_rate": 0.050},
    "WV": {"lat": 38.6, "base_temp": 52, "precip_rate": 0.100, "snow_rate": 0.025},
}


def _generate_synthetic_weather(
    states: list[str],
    start_date: date,
    end_date: date,
    seed: int = 42,
) -> list[dict]:
    """
    Generate realistic daily weather for each state.

    Physics-based approach:
        temp(day) = base + seasonal_amplitude * sin(2π * (day_of_year - 90) / 365)
                    + autocorrelated_noise
    """
    rng = np.random.default_rng(seed)
    records = []
    n_days = (end_date - start_date).days + 1

    for state in states:
        climate = _STATE_CLIMATE.get(state, {
            "lat": 39.0, "base_temp": 55, "precip_rate": 0.08, "snow_rate": 0.02,
        })

        base_temp = climate["base_temp"]
        # Seasonal amplitude scales with latitude (higher lat → bigger swings)
        seasonal_amp = 15 + (climate["lat"] - 30) * 0.8

        # Generate autocorrelated temperature noise (AR(1) process)
        noise = np.zeros(n_days)
        noise[0] = rng.normal(0, 3)
        for i in range(1, n_days):
            noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 2.5)

        for day_idx in range(n_days):
            current_date = start_date + timedelta(days=day_idx)
            doy = current_date.timetuple().tm_yday

            # Seasonal temperature curve (peaks ~July, troughs ~January)
            seasonal = seasonal_amp * math.sin(2 * math.pi * (doy - 90) / 365)
            tavg = base_temp + seasonal + noise[day_idx]

            # Diurnal range: 10-20°F, wider in dry climates
            diurnal = 12 + rng.normal(0, 2) + (0.01 - climate["precip_rate"]) * 50
            diurnal = max(5, min(25, diurnal))

            tmax = round(tavg + diurnal / 2, 1)
            tmin = round(tavg - diurnal / 2, 1)
            tavg = round(tavg, 1)

            # Precipitation: use a probability model
            precip_prob = climate["precip_rate"] * 8  # ~8-30% chance per day
            if current_date.month in (6, 7, 8) and climate["lat"] < 35:
                precip_prob *= 1.5  # summer storms in south
            if current_date.month in (11, 12, 1, 2) and climate["lat"] > 40:
                precip_prob *= 1.3  # winter precip in north

            if rng.random() < precip_prob:
                # Exponential distribution for amount (most days light rain)
                prcp = round(rng.exponential(0.3), 2)
            else:
                prcp = 0.0

            # Snow: only if cold enough
            snow = 0.0
            if tmin < 34 and prcp > 0:
                snow_ratio = climate["snow_rate"] / max(climate["precip_rate"], 0.01)
                snow = round(prcp * snow_ratio * rng.uniform(5, 15), 1)

            # Wind speed: 2-20 mph, higher in plains/coastal
            awnd = round(max(0, rng.normal(8, 3)), 1)

            records.append({
                "date": current_date.isoformat(),
                "region_id": state,
                "station_id": f"SYNTH_{state}_01",
                "tmax": tmax,
                "tmin": tmin,
                "tavg": tavg,
                "prcp": prcp,
                "snow": snow,
                "awnd": awnd,
                "source": "SYNTHETIC_NOAA_STYLE",
            })

    return records


_INSERT_SQL = """
INSERT OR REPLACE INTO raw_weather_daily
    (date, region_id, station_id, tmax, tmin, tavg, prcp, snow, awnd, source)
VALUES
    (:date, :region_id, :station_id, :tmax, :tmin, :tavg, :prcp, :snow, :awnd, :source)
"""


def ingest_weather(
    states: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    """
    Ingest weather data for the given states and date range.

    Currently uses synthetic generation. Designed so real NOAA API
    can be swapped in by implementing _fetch_noaa_weather().

    Returns:
        Summary dict with counts and timing.
    """
    if states is None:
        states = sorted(_STATE_CLIMATE.keys())
    if start_date is None:
        start_date = date(2023, 1, 1)
    if end_date is None:
        end_date = date(2025, 12, 31)

    logger.info(
        "Generating weather data for %d states, %s → %s",
        len(states), start_date, end_date,
    )

    start_time = time.time()

    # Check if NOAA API token is available
    if settings.NOAA_API_TOKEN:
        logger.info("NOAA API token found — would use real API (not implemented yet)")
        # TODO: Implement _fetch_noaa_weather() for real data
        # For now, fall through to synthetic

    records = _generate_synthetic_weather(states, start_date, end_date)

    # Clear and bulk insert
    conn = get_raw_connection()
    try:
        conn.execute("DELETE FROM raw_weather_daily;")
        batch_size = 5000
        for i in range(0, len(records), batch_size):
            conn.executemany(_INSERT_SQL, records[i:i + batch_size])
            conn.commit()
    finally:
        conn.close()

    elapsed = round(time.time() - start_time, 2)

    summary = {
        "states": len(states),
        "date_range": f"{start_date} → {end_date}",
        "days": (end_date - start_date).days + 1,
        "total_records": len(records),
        "elapsed_sec": elapsed,
        "source": "SYNTHETIC_NOAA_STYLE",
    }

    logger.info(
        "Weather ingestion complete: %d records for %d states in %.2fs",
        len(records), len(states), elapsed,
    )
    return summary


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    init_db()
    summary = ingest_weather()

    print("\n" + "=" * 60)
    print("WEATHER INGESTION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    # Sample verification
    conn = get_raw_connection()
    db_count = conn.execute("SELECT COUNT(*) FROM raw_weather_daily").fetchone()[0]
    sample = conn.execute(
        "SELECT date, region_id, tmax, tmin, prcp, snow FROM raw_weather_daily "
        "WHERE region_id = 'TX' ORDER BY date LIMIT 5"
    ).fetchall()
    temp_stats = conn.execute(
        "SELECT region_id, ROUND(AVG(tavg),1) as avg_temp, "
        "ROUND(MIN(tmin),1) as coldest, ROUND(MAX(tmax),1) as hottest "
        "FROM raw_weather_daily GROUP BY region_id ORDER BY avg_temp DESC"
    ).fetchall()
    conn.close()

    print(f"\n  DB row count: {db_count:,}")
    print(f"\n  Sample (TX, first 5 days):")
    print(f"  {'Date':12s} {'Tmax':>6s} {'Tmin':>6s} {'Prcp':>6s} {'Snow':>6s}")
    for r in sample:
        print(f"  {r[0]:12s} {r[2]:6.1f} {r[3]:6.1f} {r[4]:6.2f} {r[5]:6.1f}")

    print(f"\n  Temperature summary by state:")
    print(f"  {'State':>6s} {'Avg':>8s} {'Coldest':>8s} {'Hottest':>8s}")
    for r in temp_stats:
        print(f"  {r[0]:>6s} {r[1]:8.1f} {r[2]:8.1f} {r[3]:8.1f}")

    print("\n✅ Weather ingestion complete.")
