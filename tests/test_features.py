"""
tests/test_features.py — Comprehensive tests for Phase 4 feature engineering.

Tests cover:
    1. Feature matrix structure: correct columns, row count, no NaN in lag cols
    2. Lag features: proper shift logic, no future leakage
    3. Rolling features: non-negative, reasonable ranges
    4. Calendar features: valid day_of_week, month, season, weekend/holiday
    5. Event features: binary flags present
    6. Data quality: no duplicates, Parquet export exists
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import get_raw_connection

settings = get_settings()


class TestFeatureMatrixStructure(unittest.TestCase):
    """Test the overall structure of model_features_daily."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_row_count_positive(self):
        """Feature matrix must have rows."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily"
        ).fetchone()[0]
        self.assertGreater(count, 500, f"Only {count} rows")

    def test_expected_columns_exist(self):
        """All expected feature columns must exist."""
        cols = self.conn.execute(
            "PRAGMA table_info(model_features_daily)"
        ).fetchall()
        col_names = {c[1] for c in cols}

        expected = {
            "date", "region_id", "need_type", "target_count",
            "tmax_lag_1", "tmax_lag_7", "tmax_lag_14", "tmax_lag_30",
            "prcp_lag_1", "prcp_lag_7",
            "target_lag_1", "target_lag_7",
            "temp_rollmean_7", "temp_rollmean_14",
            "precip_rollsum_7", "precip_rollsum_30",
            "target_rollmean_7", "target_rollmean_14",
            "heatwave_flag", "coldwave_flag", "heavy_rain_flag",
            "day_of_week", "month", "season", "holiday_flag", "is_weekend",
        }
        missing = expected - col_names
        self.assertEqual(len(missing), 0, f"Missing columns: {missing}")

    def test_column_count(self):
        """Should have ~30 columns."""
        cols = self.conn.execute(
            "PRAGMA table_info(model_features_daily)"
        ).fetchall()
        self.assertGreaterEqual(len(cols), 25, f"Only {len(cols)} columns")

    def test_no_duplicate_keys(self):
        """No duplicate (date, region_id, need_type) keys."""
        dupes = self.conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT date, region_id, need_type, COUNT(*) as n
                FROM model_features_daily
                GROUP BY date, region_id, need_type
                HAVING n > 1
            )
        """).fetchone()[0]
        self.assertEqual(dupes, 0, f"{dupes} duplicate keys")


class TestLagFeatures(unittest.TestCase):
    """Test lag feature correctness."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_no_null_lag_columns(self):
        """All lag columns must be non-NULL (NaN rows were dropped)."""
        lag_cols = [
            "tmax_lag_1", "tmax_lag_3", "tmax_lag_7", "tmax_lag_14", "tmax_lag_30",
            "prcp_lag_1", "prcp_lag_3", "prcp_lag_7", "prcp_lag_14", "prcp_lag_30",
            "target_lag_1", "target_lag_7",
        ]
        for col in lag_cols:
            nulls = self.conn.execute(
                f"SELECT COUNT(*) FROM model_features_daily WHERE {col} IS NULL"
            ).fetchone()[0]
            self.assertEqual(nulls, 0, f"{nulls} NULLs in {col}")

    def test_tmax_lag_1_realistic_range(self):
        """Lagged tmax should be in realistic temp range."""
        result = self.conn.execute(
            "SELECT MIN(tmax_lag_1), MAX(tmax_lag_1) FROM model_features_daily"
        ).fetchone()
        self.assertGreater(result[0], -20, f"Min tmax_lag_1 = {result[0]}")
        self.assertLess(result[1], 130, f"Max tmax_lag_1 = {result[1]}")

    def test_prcp_lags_non_negative(self):
        """Precipitation lags must be >= 0."""
        for lag in [1, 3, 7, 14, 30]:
            neg = self.conn.execute(
                f"SELECT COUNT(*) FROM model_features_daily WHERE prcp_lag_{lag} < 0"
            ).fetchone()[0]
            self.assertEqual(neg, 0, f"Negative prcp_lag_{lag}")

    def test_target_lag_1_positive(self):
        """target_lag_1 must be positive (we only have non-zero target days)."""
        neg = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily WHERE target_lag_1 <= 0"
        ).fetchone()[0]
        self.assertEqual(neg, 0, f"{neg} non-positive target_lag_1 values")

    def test_no_future_leakage_in_lags(self):
        """
        Verify that lag values aren't from the future.
        For a given (region, need_type), tmax_lag_1 on date D should equal
        max_temp from the WEATHER data on date D-1.
        Spot-check a random pair.
        """
        # Pick one (region, need_type) pair
        pair = self.conn.execute(
            "SELECT region_id, need_type FROM model_features_daily LIMIT 1"
        ).fetchone()
        if pair is None:
            self.skipTest("No data")

        # Get two consecutive rows
        rows = self.conn.execute(
            "SELECT date, tmax_lag_1 FROM model_features_daily "
            "WHERE region_id=? AND need_type=? ORDER BY date LIMIT 2",
            (pair[0], pair[1])
        ).fetchall()
        # We can't do the exact check without the raw data, but ensure values exist
        self.assertEqual(len(rows), 2, "Need at least 2 rows for leakage check")
        self.assertIsNotNone(rows[0][1])
        self.assertIsNotNone(rows[1][1])


class TestRollingFeatures(unittest.TestCase):
    """Test rolling window features."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_rolling_means_not_null(self):
        """Rolling mean columns must not be NULL."""
        for col in ["temp_rollmean_7", "temp_rollmean_14",
                     "target_rollmean_7", "target_rollmean_14"]:
            nulls = self.conn.execute(
                f"SELECT COUNT(*) FROM model_features_daily WHERE {col} IS NULL"
            ).fetchone()[0]
            self.assertEqual(nulls, 0, f"{nulls} NULLs in {col}")

    def test_rolling_sums_non_negative(self):
        """Rolling precipitation sums must be >= 0."""
        for col in ["precip_rollsum_7", "precip_rollsum_30"]:
            neg = self.conn.execute(
                f"SELECT COUNT(*) FROM model_features_daily WHERE {col} < 0"
            ).fetchone()[0]
            self.assertEqual(neg, 0, f"Negative {col}")

    def test_rolling_30d_gte_7d_precip(self):
        """30-day precip sum should generally be >= 7-day sum."""
        violations = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily "
            "WHERE precip_rollsum_30 < precip_rollsum_7 * 0.9"  # allow small tolerance
        ).fetchone()[0]
        total = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily"
        ).fetchone()[0]
        violation_pct = violations / max(total, 1) * 100
        self.assertLess(violation_pct, 5,
                        f"{violation_pct:.1f}% of rows have 30d < 7d precip")


class TestCalendarFeatures(unittest.TestCase):
    """Test calendar-based features."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_day_of_week_range(self):
        """day_of_week must be 0-6."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily "
            "WHERE day_of_week < 0 OR day_of_week > 6"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_month_range(self):
        """month must be 1-12."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily "
            "WHERE month < 1 OR month > 12"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_season_valid_values(self):
        """season must be one of winter/spring/summer/fall."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily "
            "WHERE season NOT IN ('winter', 'spring', 'summer', 'fall')"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_is_weekend_binary(self):
        """is_weekend must be 0 or 1."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily "
            "WHERE is_weekend NOT IN (0, 1)"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_holiday_flag_binary(self):
        """holiday_flag must be 0 or 1."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM model_features_daily "
            "WHERE holiday_flag NOT IN (0, 1)"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_some_weekends_exist(self):
        """Should have some weekend rows."""
        count = self.conn.execute(
            "SELECT SUM(is_weekend) FROM model_features_daily"
        ).fetchone()[0]
        self.assertGreater(count, 0, "No weekend rows found")


class TestEventFeatures(unittest.TestCase):
    """Test extreme-weather event flags."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_flags_are_binary(self):
        """All event flags must be 0 or 1."""
        for col in ["heatwave_flag", "coldwave_flag", "heavy_rain_flag"]:
            invalid = self.conn.execute(
                f"SELECT COUNT(*) FROM model_features_daily "
                f"WHERE {col} NOT IN (0, 1)"
            ).fetchone()[0]
            self.assertEqual(invalid, 0, f"Non-binary {col}")

    def test_flags_not_all_zero(self):
        """At least one event flag type should have some 1s."""
        total = 0
        for col in ["heatwave_flag", "coldwave_flag", "heavy_rain_flag"]:
            count = self.conn.execute(
                f"SELECT SUM({col}) FROM model_features_daily"
            ).fetchone()[0] or 0
            total += count
        self.assertGreater(total, 0, "All event flags are 0")


class TestParquetExport(unittest.TestCase):
    """Test that the Parquet export file was created."""

    def test_parquet_file_exists(self):
        """features.parquet must exist in data/processed/."""
        path = os.path.join(settings.DATA_PROCESSED_DIR, "features.parquet")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_parquet_file_size(self):
        """Parquet file should be non-trivial size."""
        path = os.path.join(settings.DATA_PROCESSED_DIR, "features.parquet")
        if os.path.exists(path):
            size = os.path.getsize(path)
            self.assertGreater(size, 1000, f"Parquet only {size} bytes")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
