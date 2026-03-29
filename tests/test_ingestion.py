"""
tests/test_ingestion.py — Comprehensive tests for Phase 2 data ingestion.

Tests cover:
    1. Region lookup construction and correctness
    2. Social needs ingestion: schema handling, row counts, categories, dates
    3. Weather ingestion: synthetic generation quality, temperature physics, coverage
    4. Cross-table alignment: every social-needs region has matching weather
    5. Data quality: no NULLs in critical columns, valid ranges, no duplicates
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import get_raw_connection, init_db


class TestDatabaseSetup(unittest.TestCase):
    """Verify the database is properly initialized."""

    def test_all_tables_exist(self):
        """All 9 expected tables must exist."""
        conn = get_raw_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        conn.close()

        table_names = {t[0] for t in tables}
        expected = {
            "region_lookup",
            "raw_weather_daily",
            "raw_social_needs",
            "weather_daily_agg",
            "social_needs_daily_agg",
            "model_features_daily",
            "experiments",
            "metrics",
            "causal_results",
        }
        for t in expected:
            self.assertIn(t, table_names, f"Missing table: {t}")

    def test_indexes_exist(self):
        """Critical indexes must be created."""
        conn = get_raw_connection()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        conn.close()

        index_names = {i[0] for i in indexes}
        expected_prefixes = [
            "idx_raw_wx_date_region",
            "idx_raw_sn_date",
            "idx_raw_sn_category",
            "idx_region_zip",
        ]
        for prefix in expected_prefixes:
            self.assertTrue(
                any(prefix in name for name in index_names),
                f"Missing index with prefix: {prefix}",
            )


class TestRegionLookup(unittest.TestCase):
    """Test the region lookup table."""

    def test_region_lookup_populated(self):
        """region_lookup must have data."""
        conn = get_raw_connection()
        count = conn.execute("SELECT COUNT(*) FROM region_lookup").fetchone()[0]
        conn.close()
        self.assertGreater(count, 0, "region_lookup is empty")

    def test_all_24_states_present(self):
        """All 24 states from the GroundGame data must be represented."""
        expected_states = {
            "AZ", "CA", "CO", "CT", "DC", "FL", "GA", "IA", "IN", "KY",
            "LA", "ME", "MO", "NH", "NJ", "NV", "NY", "OH", "TN", "TX",
            "VA", "WA", "WI", "WV",
        }
        conn = get_raw_connection()
        states = conn.execute(
            "SELECT DISTINCT state FROM region_lookup"
        ).fetchall()
        conn.close()

        actual_states = {s[0] for s in states}
        missing = expected_states - actual_states
        self.assertEqual(len(missing), 0, f"Missing states: {missing}")

    def test_zip_codes_reasonable_count(self):
        """Should have thousands of unique ZIP codes."""
        conn = get_raw_connection()
        count = conn.execute(
            "SELECT COUNT(DISTINCT zipcode) FROM region_lookup"
        ).fetchone()[0]
        conn.close()
        self.assertGreater(count, 5000, f"Only {count} ZIPs — expected 5000+")

    def test_no_null_states(self):
        """No NULL states allowed in region_lookup."""
        conn = get_raw_connection()
        nulls = conn.execute(
            "SELECT COUNT(*) FROM region_lookup WHERE state IS NULL OR state = ''"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(nulls, 0, f"Found {nulls} NULL/empty states")


class TestSocialNeedsIngestion(unittest.TestCase):
    """Test the social needs data in raw_social_needs."""

    def test_row_count(self):
        """Should have ingested 100K+ rows across the 2 CSV files."""
        conn = get_raw_connection()
        count = conn.execute("SELECT COUNT(*) FROM raw_social_needs").fetchone()[0]
        conn.close()
        self.assertGreater(count, 100_000, f"Only {count} rows — expected 100K+")

    def test_all_16_categories_present(self):
        """All 16 need categories from GroundGame must be present."""
        expected_categories = {
            "Food Insecurity", "Housing Insecurity", "Financial Hardship",
            "Utility Insecurity", "Transportation Insecurity", "Clinical Barriers",
            "Employment Insecurity", "Digital Inequity", "Literacy",
            "Caregiver Support", "Interpersonal Safety", "Physical Safety",
            "Legal Hardship", "Social Inclusion", "Health Benefits",
            "Other Assistance",
        }
        conn = get_raw_connection()
        cats = conn.execute(
            "SELECT DISTINCT category FROM raw_social_needs"
        ).fetchall()
        conn.close()

        actual = {c[0] for c in cats}
        missing = expected_categories - actual
        self.assertEqual(len(missing), 0, f"Missing categories: {missing}")

    def test_date_range(self):
        """Date range should span Jan to Jul 2025."""
        conn = get_raw_connection()
        result = conn.execute(
            "SELECT MIN(ref_date), MAX(ref_date) FROM raw_social_needs"
        ).fetchone()
        conn.close()

        min_date = result[0][:10]
        max_date = result[1][:10]
        self.assertTrue(min_date.startswith("2025-01"), f"Min date: {min_date}")
        self.assertTrue(max_date.startswith("2025-07"), f"Max date: {max_date}")

    def test_no_null_category(self):
        """category column must never be NULL."""
        conn = get_raw_connection()
        nulls = conn.execute(
            "SELECT COUNT(*) FROM raw_social_needs WHERE category IS NULL OR category = ''"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(nulls, 0, f"Found {nulls} NULL categories")

    def test_no_null_ref_date(self):
        """ref_date must never be NULL."""
        conn = get_raw_connection()
        nulls = conn.execute(
            "SELECT COUNT(*) FROM raw_social_needs WHERE ref_date IS NULL OR ref_date = ''"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(nulls, 0, f"Found {nulls} NULL ref_dates")

    def test_states_match_region_lookup(self):
        """Every state in raw_social_needs should exist in region_lookup."""
        conn = get_raw_connection()
        orphan_states = conn.execute("""
            SELECT DISTINCT s.state FROM raw_social_needs s
            WHERE s.state IS NOT NULL AND s.state != ''
            AND s.state NOT IN (SELECT DISTINCT state FROM region_lookup)
        """).fetchall()
        conn.close()
        self.assertEqual(len(orphan_states), 0,
                         f"Orphan states: {[s[0] for s in orphan_states]}")

    def test_need_statuses_valid(self):
        """Need statuses should only be known values."""
        valid = {"Confirmed", "Unconfirmed", "Unmet", "InProcess", "YetToStart", ""}
        conn = get_raw_connection()
        statuses = conn.execute(
            "SELECT DISTINCT need_status FROM raw_social_needs"
        ).fetchall()
        conn.close()

        actual = {s[0] for s in statuses if s[0]}
        unexpected = actual - valid
        self.assertEqual(len(unexpected), 0, f"Unexpected statuses: {unexpected}")

    def test_source_files_tracked(self):
        """Each row should track which CSV it came from."""
        conn = get_raw_connection()
        sources = conn.execute(
            "SELECT DISTINCT source_file FROM raw_social_needs"
        ).fetchall()
        conn.close()

        source_names = {s[0] for s in sources}
        self.assertGreaterEqual(len(source_names), 2, "Expected at least 2 source files")


class TestWeatherIngestion(unittest.TestCase):
    """Test the weather data in raw_weather_daily."""

    def test_row_count(self):
        """Should have 24 states × ~212 days = ~5,088 rows."""
        conn = get_raw_connection()
        count = conn.execute("SELECT COUNT(*) FROM raw_weather_daily").fetchone()[0]
        conn.close()
        self.assertGreater(count, 4000, f"Only {count} weather rows — expected 4000+")

    def test_all_24_states_have_weather(self):
        """Every state must have weather data."""
        expected_states = {
            "AZ", "CA", "CO", "CT", "DC", "FL", "GA", "IA", "IN", "KY",
            "LA", "ME", "MO", "NH", "NJ", "NV", "NY", "OH", "TN", "TX",
            "VA", "WA", "WI", "WV",
        }
        conn = get_raw_connection()
        states = conn.execute(
            "SELECT DISTINCT region_id FROM raw_weather_daily"
        ).fetchall()
        conn.close()

        actual = {s[0] for s in states}
        missing = expected_states - actual
        self.assertEqual(len(missing), 0, f"Missing weather for states: {missing}")

    def test_date_range_matches_social_needs(self):
        """Weather date range should cover the social-needs date range."""
        conn = get_raw_connection()
        wx_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM raw_weather_daily"
        ).fetchone()
        sn_range = conn.execute(
            "SELECT MIN(ref_date), MAX(ref_date) FROM raw_social_needs"
        ).fetchone()
        conn.close()

        self.assertLessEqual(wx_range[0], sn_range[0][:10],
                             "Weather starts after social needs")
        self.assertGreaterEqual(wx_range[1], sn_range[1][:10],
                                "Weather ends before social needs")

    def test_temperature_ranges_realistic(self):
        """Temperatures should be in a realistic range (-30°F to 130°F)."""
        conn = get_raw_connection()
        result = conn.execute(
            "SELECT MIN(tmin), MAX(tmax) FROM raw_weather_daily"
        ).fetchone()
        conn.close()

        self.assertGreater(result[0], -30, f"Unrealistic tmin: {result[0]}")
        self.assertLess(result[1], 130, f"Unrealistic tmax: {result[1]}")

    def test_tmax_always_greater_than_tmin(self):
        """Max temperature must always exceed min temperature."""
        conn = get_raw_connection()
        violations = conn.execute(
            "SELECT COUNT(*) FROM raw_weather_daily WHERE tmax < tmin"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(violations, 0, f"{violations} rows where tmax < tmin")

    def test_precipitation_non_negative(self):
        """Precipitation must be >= 0."""
        conn = get_raw_connection()
        negatives = conn.execute(
            "SELECT COUNT(*) FROM raw_weather_daily WHERE prcp < 0"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(negatives, 0, f"{negatives} rows with negative precipitation")

    def test_florida_hotter_than_maine(self):
        """Florida should have higher average temperature than Maine."""
        conn = get_raw_connection()
        fl_avg = conn.execute(
            "SELECT AVG(tavg) FROM raw_weather_daily WHERE region_id='FL'"
        ).fetchone()[0]
        me_avg = conn.execute(
            "SELECT AVG(tavg) FROM raw_weather_daily WHERE region_id='ME'"
        ).fetchone()[0]
        conn.close()

        self.assertGreater(fl_avg, me_avg,
                           f"FL avg {fl_avg:.1f} should be > ME avg {me_avg:.1f}")

    def test_summer_hotter_than_winter(self):
        """July should be hotter than January for any northern state."""
        conn = get_raw_connection()
        jan_avg = conn.execute(
            "SELECT AVG(tavg) FROM raw_weather_daily "
            "WHERE region_id='NY' AND date LIKE '2025-01%'"
        ).fetchone()[0]
        jul_avg = conn.execute(
            "SELECT AVG(tavg) FROM raw_weather_daily "
            "WHERE region_id='NY' AND date LIKE '2025-07%'"
        ).fetchone()[0]
        conn.close()

        self.assertGreater(jul_avg, jan_avg,
                           f"NY Jul avg {jul_avg:.1f} should be > Jan avg {jan_avg:.1f}")

    def test_no_null_critical_columns(self):
        """date, region_id, tmax, tmin must never be NULL."""
        conn = get_raw_connection()
        for col in ["date", "region_id", "tmax", "tmin"]:
            nulls = conn.execute(
                f"SELECT COUNT(*) FROM raw_weather_daily WHERE {col} IS NULL"
            ).fetchone()[0]
            self.assertEqual(nulls, 0, f"Found {nulls} NULL values in {col}")
        conn.close()

    def test_snow_only_when_cold(self):
        """Snow should only occur when tmin < 40°F."""
        conn = get_raw_connection()
        violations = conn.execute(
            "SELECT COUNT(*) FROM raw_weather_daily WHERE snow > 0 AND tmin > 40"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(violations, 0,
                         f"{violations} rows with snow when tmin > 40°F")


class TestCrossTableAlignment(unittest.TestCase):
    """Test alignment between weather and social-needs data."""

    def test_social_needs_states_have_weather(self):
        """Every state in social needs must have weather data."""
        conn = get_raw_connection()
        orphans = conn.execute("""
            SELECT DISTINCT s.state FROM raw_social_needs s
            WHERE s.state IS NOT NULL AND s.state != ''
            AND s.state NOT IN (SELECT DISTINCT region_id FROM raw_weather_daily)
        """).fetchall()
        conn.close()
        self.assertEqual(len(orphans), 0,
                         f"States without weather: {[o[0] for o in orphans]}")

    def test_date_overlap(self):
        """Weather and social needs date ranges must overlap."""
        conn = get_raw_connection()
        wx_min = conn.execute("SELECT MIN(date) FROM raw_weather_daily").fetchone()[0]
        wx_max = conn.execute("SELECT MAX(date) FROM raw_weather_daily").fetchone()[0]
        sn_min = conn.execute("SELECT MIN(ref_date) FROM raw_social_needs").fetchone()[0][:10]
        sn_max = conn.execute("SELECT MAX(ref_date) FROM raw_social_needs").fetchone()[0][:10]
        conn.close()

        # Ranges must overlap
        self.assertTrue(wx_min <= sn_max and sn_min <= wx_max,
                        f"No date overlap: weather [{wx_min},{wx_max}] vs social [{sn_min},{sn_max}]")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
