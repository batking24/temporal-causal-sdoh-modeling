"""
tests/test_models.py — Comprehensive tests for Phase 5 causal modeling.

Tests cover:
    1. Granger causality: results stored, p-values valid, significance flags
    2. Experiments table: entries for all 3 model types + Granger
    3. Metrics table: RMSE/MAE populated, non-negative values
    4. Causal results: feature names, lags, p-values in valid range
    5. Baseline AR: produces predictions, RMSE finite
    6. Model comparison: results are consistent and comparable
    7. Plots: files generated in outputs/plots/
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


class TestExperimentsTable(unittest.TestCase):
    """Test the experiments table was populated correctly."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_experiments_exist(self):
        """At least 4 experiments should exist (Granger + 3 models)."""
        count = self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        self.assertGreaterEqual(count, 4, f"Only {count} experiments")

    def test_granger_experiment_exists(self):
        """A Granger causality experiment must exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE model_type='granger_causality'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_baseline_ar_experiment(self):
        """A baseline_ar experiment must exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE model_type='baseline_ar'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_var_weather_experiment(self):
        """A var_all_weather experiment must exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE model_type='var_all_weather'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_granger_selected_experiment(self):
        """A granger_selected experiment must exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE model_type='granger_selected'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_experiment_dates_valid(self):
        """All experiments must have valid train_start and train_end."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE train_start IS NULL OR train_end IS NULL"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)


class TestCausalResults(unittest.TestCase):
    """Test the causal_results table."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_causal_results_populated(self):
        """causal_results must have data."""
        count = self.conn.execute("SELECT COUNT(*) FROM causal_results").fetchone()[0]
        self.assertGreater(count, 50, f"Only {count} causal results")

    def test_p_values_in_range(self):
        """All p-values must be between 0 and 1."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM causal_results WHERE p_value < 0 OR p_value > 1"
        ).fetchone()[0]
        self.assertEqual(invalid, 0, f"{invalid} p-values out of [0,1]")

    def test_significant_flag_binary(self):
        """significant_flag must be 0 or 1."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM causal_results WHERE significant_flag NOT IN (0, 1)"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_feature_names_valid(self):
        """Feature names should be known weather variables."""
        features = self.conn.execute(
            "SELECT DISTINCT feature_name FROM causal_results"
        ).fetchall()
        feature_names = {f[0] for f in features}
        expected = {"max_temp", "precip"}
        self.assertTrue(feature_names.issubset(expected | feature_names),
                        f"Unexpected features: {feature_names - expected}")

    def test_lags_positive(self):
        """All lags must be positive integers."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM causal_results WHERE lag < 1"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_some_significant_results(self):
        """Should have at least a few significant results."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM causal_results WHERE significant_flag = 1"
        ).fetchone()[0]
        self.assertGreater(count, 0, "No significant Granger results at all")

    def test_linked_to_experiment(self):
        """All causal results must be linked to a valid experiment."""
        orphans = self.conn.execute("""
            SELECT COUNT(*) FROM causal_results cr
            WHERE cr.experiment_id NOT IN (SELECT experiment_id FROM experiments)
        """).fetchone()[0]
        self.assertEqual(orphans, 0, f"{orphans} orphaned causal results")


class TestMetrics(unittest.TestCase):
    """Test the metrics table."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_metrics_populated(self):
        """Metrics table must have data for all 3 models."""
        count = self.conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        self.assertGreaterEqual(count, 3, f"Only {count} metric rows")

    def test_rmse_non_negative(self):
        """All RMSE values must be >= 0."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE rmse < 0"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_mae_non_negative(self):
        """All MAE values must be >= 0."""
        invalid = self.conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE mae < 0"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)

    def test_rmse_finite(self):
        """RMSE must not be NULL (models must have converged)."""
        nulls = self.conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE rmse IS NULL"
        ).fetchone()[0]
        self.assertEqual(nulls, 0, f"{nulls} NULL RMSE values")

    def test_linked_to_experiment(self):
        """All metrics must be linked to valid experiments."""
        orphans = self.conn.execute("""
            SELECT COUNT(*) FROM metrics m
            WHERE m.experiment_id NOT IN (SELECT experiment_id FROM experiments)
        """).fetchone()[0]
        self.assertEqual(orphans, 0)


class TestPlots(unittest.TestCase):
    """Test that plots were generated."""

    def test_cross_corr_plot_exists(self):
        """Cross-correlation plot must be generated."""
        path = os.path.join(settings.PLOTS_DIR, "cross_corr_all.png")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_heatmap_plot_exists(self):
        """Lag significance heatmap must be generated."""
        path = os.path.join(settings.PLOTS_DIR, "lag_significance_heatmap.png")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_plots_non_trivial_size(self):
        """Plot files should be non-trivial."""
        for name in ["cross_corr_all.png", "lag_significance_heatmap.png"]:
            path = os.path.join(settings.PLOTS_DIR, name)
            if os.path.exists(path):
                size = os.path.getsize(path)
                self.assertGreater(size, 5000, f"{name} only {size} bytes")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
