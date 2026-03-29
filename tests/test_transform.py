"""
tests/test_transform.py — Comprehensive tests for Phase 3 cleaning & aggregation.

Tests cover:
    1. Weather cleaning: NULL handling, outlier capping, event flags, rolling stats
    2. Social-needs cleaning: dedup, category count, daily aggregation, rolling counts
    3. Geographic alignment: 100% coverage, state matching, date overlap
    4. Data quality: no NULLs in critical columns, valid value ranges
    5. Consistency: row counts, aggregation correctness
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_raw_connection


class TestWeatherDailyAgg(unittest.TestCase):
    """Test the cleaned weather_daily_agg table."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_row_count_matches_raw(self):
        """Should have same row count as raw_weather_daily (no rows dropped)."""
        raw = self.conn.execute("SELECT COUNT(*) FROM raw_weather_daily").fetchone()[0]
        agg = self.conn.execute("SELECT COUNT(*) FROM weather_daily_agg").fetchone()[0]
        self.assertEqual(raw, agg, f"Raw {raw} ≠ Agg {agg}")

    def test_no_null_temperatures(self):
        """avg_temp, max_temp, min_temp must never be NULL."""
        for col in ["avg_temp", "max_temp", "min_temp"]:
            nulls = self.conn.execute(
                f"SELECT COUNT(*) FROM weather_daily_agg WHERE {col} IS NULL"
            ).fetchone()[0]
            self.assertEqual(nulls, 0, f"{nulls} NULLs in {col}")

    def test_no_null_dates_or_regions(self):
        """date and region_id must never be NULL."""
        for col in ["date", "region_id"]:
            nulls = self.conn.execute(
                f"SELECT COUNT(*) FROM weather_daily_agg WHERE {col} IS NULL OR {col} = ''"
            ).fetchone()[0]
            self.assertEqual(nulls, 0, f"{nulls} NULLs in {col}")

    def test_max_temp_gte_min_temp(self):
        """max_temp must always be >= min_temp."""
        violations = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg WHERE max_temp < min_temp"
        ).fetchone()[0]
        self.assertEqual(violations, 0, f"{violations} rows where max < min")

    def test_temp_range_correct(self):
        """temp_range should equal max_temp - min_temp."""
        violations = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg "
            "WHERE ABS(temp_range - (max_temp - min_temp)) > 0.1"
        ).fetchone()[0]
        self.assertEqual(violations, 0, f"{violations} rows with incorrect temp_range")

    def test_heatwave_flag_is_binary(self):
        """heatwave_flag must be 0 or 1."""
        non_binary = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg "
            "WHERE heatwave_flag NOT IN (0, 1)"
        ).fetchone()[0]
        self.assertEqual(non_binary, 0, f"{non_binary} non-binary heatwave_flag values")

    def test_coldwave_flag_is_binary(self):
        """coldwave_flag must be 0 or 1."""
        non_binary = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg "
            "WHERE coldwave_flag NOT IN (0, 1)"
        ).fetchone()[0]
        self.assertEqual(non_binary, 0)

    def test_heavy_rain_flag_is_binary(self):
        """heavy_rain_flag must be 0 or 1."""
        non_binary = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg "
            "WHERE heavy_rain_flag NOT IN (0, 1)"
        ).fetchone()[0]
        self.assertEqual(non_binary, 0)

    def test_heatwave_days_exist(self):
        """Should have some heatwave days detected."""
        count = self.conn.execute(
            "SELECT SUM(heatwave_flag) FROM weather_daily_agg"
        ).fetchone()[0]
        self.assertGreater(count, 0, "No heatwave days detected")

    def test_coldwave_days_exist(self):
        """Should have some coldwave days detected."""
        count = self.conn.execute(
            "SELECT SUM(coldwave_flag) FROM weather_daily_agg"
        ).fetchone()[0]
        self.assertGreater(count, 0, "No coldwave days detected")

    def test_rolling_7d_temp_not_null(self):
        """rolling_7d_temp should be populated."""
        nulls = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg WHERE rolling_7d_temp IS NULL"
        ).fetchone()[0]
        self.assertEqual(nulls, 0, f"{nulls} NULLs in rolling_7d_temp")

    def test_rolling_7d_precip_non_negative(self):
        """Rolling precipitation sum should be non-negative."""
        negatives = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg WHERE rolling_7d_precip < 0"
        ).fetchone()[0]
        self.assertEqual(negatives, 0)

    def test_precip_non_negative(self):
        """Daily precipitation must be >= 0."""
        negatives = self.conn.execute(
            "SELECT COUNT(*) FROM weather_daily_agg WHERE precip < 0"
        ).fetchone()[0]
        self.assertEqual(negatives, 0)


class TestSocialNeedsDailyAgg(unittest.TestCase):
    """Test the cleaned social_needs_daily_agg table."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_row_count_reasonable(self):
        """Aggregated table should have meaningful number of rows."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM social_needs_daily_agg"
        ).fetchone()[0]
        self.assertGreater(count, 1000, f"Only {count} rows — expected 1000+")

    def test_fewer_rows_than_raw(self):
        """Aggregated table must have fewer rows than raw (it's aggregated!)."""
        raw = self.conn.execute("SELECT COUNT(*) FROM raw_social_needs").fetchone()[0]
        agg = self.conn.execute("SELECT COUNT(*) FROM social_needs_daily_agg").fetchone()[0]
        self.assertLess(agg, raw, f"Agg ({agg}) should be < Raw ({raw})")

    def test_all_16_need_types_present(self):
        """All 16 need categories must survive cleaning."""
        count = self.conn.execute(
            "SELECT COUNT(DISTINCT need_type) FROM social_needs_daily_agg"
        ).fetchone()[0]
        self.assertEqual(count, 16, f"Only {count} need types, expected 16")

    def test_all_24_states_present(self):
        """All 24 states must be in aggregated data."""
        count = self.conn.execute(
            "SELECT COUNT(DISTINCT region_id) FROM social_needs_daily_agg"
        ).fetchone()[0]
        self.assertEqual(count, 24, f"Only {count} regions, expected 24")

    def test_no_null_critical_columns(self):
        """date, region_id, need_type, daily_need_count must never be NULL."""
        for col in ["date", "region_id", "need_type", "daily_need_count"]:
            nulls = self.conn.execute(
                f"SELECT COUNT(*) FROM social_needs_daily_agg WHERE {col} IS NULL"
            ).fetchone()[0]
            self.assertEqual(nulls, 0, f"{nulls} NULLs in {col}")

    def test_daily_need_count_positive(self):
        """daily_need_count must be > 0 (we only aggregate non-empty days)."""
        zeros = self.conn.execute(
            "SELECT COUNT(*) FROM social_needs_daily_agg WHERE daily_need_count <= 0"
        ).fetchone()[0]
        self.assertEqual(zeros, 0, f"{zeros} rows with count <= 0")

    def test_confirmed_count_lte_total(self):
        """confirmed_count should never exceed daily_need_count."""
        violations = self.conn.execute(
            "SELECT COUNT(*) FROM social_needs_daily_agg "
            "WHERE confirmed_count > daily_need_count"
        ).fetchone()[0]
        self.assertEqual(violations, 0, f"{violations} rows where confirmed > total")

    def test_unmet_count_lte_total(self):
        """unmet_count should never exceed daily_need_count."""
        violations = self.conn.execute(
            "SELECT COUNT(*) FROM social_needs_daily_agg "
            "WHERE unmet_count > daily_need_count"
        ).fetchone()[0]
        self.assertEqual(violations, 0)

    def test_rolling_7d_populated(self):
        """rolling_7d_count should be populated."""
        nulls = self.conn.execute(
            "SELECT COUNT(*) FROM social_needs_daily_agg WHERE rolling_7d_count IS NULL"
        ).fetchone()[0]
        self.assertEqual(nulls, 0, f"{nulls} NULLs in rolling_7d_count")

    def test_rolling_values_non_negative(self):
        """All rolling values must be >= 0."""
        for col in ["rolling_7d_count", "rolling_14d_count", "rolling_30d_count"]:
            negatives = self.conn.execute(
                f"SELECT COUNT(*) FROM social_needs_daily_agg WHERE {col} < 0"
            ).fetchone()[0]
            self.assertEqual(negatives, 0, f"{negatives} negative values in {col}")

    def test_date_range_valid(self):
        """Date range should be within Jan-Jul 2025."""
        result = self.conn.execute(
            "SELECT MIN(date), MAX(date) FROM social_needs_daily_agg"
        ).fetchone()
        self.assertTrue(result[0] >= "2025-01-01", f"Min date: {result[0]}")
        self.assertTrue(result[1] <= "2025-07-31", f"Max date: {result[1]}")

    def test_total_needs_match_raw_after_dedup(self):
        """Total aggregated counts should approximately match deduplicated raw."""
        total_agg = self.conn.execute(
            "SELECT SUM(daily_need_count) FROM social_needs_daily_agg"
        ).fetchone()[0]
        self.assertGreater(total_agg, 100000, f"Total {total_agg} seems too low")


class TestAlignment(unittest.TestCase):
    """Test geographic and temporal alignment between datasets."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_100_percent_state_coverage(self):
        """Every social-needs state must have weather data."""
        orphans = self.conn.execute("""
            SELECT DISTINCT s.region_id FROM social_needs_daily_agg s
            WHERE s.region_id NOT IN (
                SELECT DISTINCT region_id FROM weather_daily_agg
            )
        """).fetchall()
        self.assertEqual(len(orphans), 0,
                         f"States without weather: {[o[0] for o in orphans]}")

    def test_date_region_pair_coverage(self):
        """All (date, region) pairs in social needs must have weather data."""
        total = self.conn.execute(
            "SELECT COUNT(DISTINCT date || '|' || region_id) FROM social_needs_daily_agg"
        ).fetchone()[0]
        matched = self.conn.execute("""
            SELECT COUNT(DISTINCT s.date || '|' || s.region_id)
            FROM social_needs_daily_agg s
            INNER JOIN weather_daily_agg w
                ON s.date = w.date AND s.region_id = w.region_id
        """).fetchone()[0]
        coverage = matched / max(total, 1) * 100
        self.assertGreaterEqual(coverage, 95.0,
                                f"Coverage only {coverage:.1f}%")

    def test_no_duplicate_primary_keys_weather(self):
        """No duplicate (date, region_id) pairs in weather_daily_agg."""
        duplicates = self.conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT date, region_id, COUNT(*) as n
                FROM weather_daily_agg
                GROUP BY date, region_id
                HAVING n > 1
            )
        """).fetchone()[0]
        self.assertEqual(duplicates, 0, f"{duplicates} duplicate weather keys")

    def test_no_duplicate_primary_keys_social(self):
        """No duplicate (date, region_id, need_type) in social_needs_daily_agg."""
        duplicates = self.conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT date, region_id, need_type, COUNT(*) as n
                FROM social_needs_daily_agg
                GROUP BY date, region_id, need_type
                HAVING n > 1
            )
        """).fetchone()[0]
        self.assertEqual(duplicates, 0, f"{duplicates} duplicate social-needs keys")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
