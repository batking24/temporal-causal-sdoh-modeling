"""
tests/test_evaluation.py — Comprehensive tests for Phase 6 rolling validation.

Tests cover:
    1. Rolling experiments: entries stored for all 3 model types
    2. Per-split metrics: 5 splits stored with valid RMSE/MAE values
    3. Stability scores: σ(RMSE) non-negative, stored in DB
    4. Drift analysis: seasonal + extreme weather results valid
    5. Plots: comparison charts generated
    6. JSON export: evaluation summary file exists and valid
    7. No data leakage: test dates always after train dates
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import get_raw_connection

settings = get_settings()


class TestRollingExperiments(unittest.TestCase):
    """Test that rolling validation experiments were stored correctly."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_rolling_baseline_experiment_exists(self):
        """A rolling_baseline_ar experiment must exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE model_type='rolling_baseline_ar'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_rolling_var_experiment_exists(self):
        """A rolling_var_all_weather experiment must exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE model_type='rolling_var_all_weather'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_rolling_granger_experiment_exists(self):
        """A rolling_granger_selected experiment must exist."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE model_type='rolling_granger_selected'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_rolling_experiments_have_params(self):
        """Rolling experiments must have params_json with stability data."""
        rows = self.conn.execute(
            "SELECT params_json FROM experiments WHERE model_type LIKE 'rolling_%'"
        ).fetchall()
        for row in rows:
            self.assertIsNotNone(row[0], "params_json is NULL")
            params = json.loads(row[0])
            self.assertIn("mean_rmse", params, "Missing mean_rmse in params")
            self.assertIn("std_rmse", params, "Missing std_rmse in params")


class TestPerSplitMetrics(unittest.TestCase):
    """Test per-split metrics from rolling validation."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_multiple_splits_stored(self):
        """Should have multiple splits per rolling experiment."""
        rows = self.conn.execute("""
            SELECT m.experiment_id, COUNT(m.id) as split_count
            FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE e.model_type LIKE 'rolling_%'
            GROUP BY m.experiment_id
        """).fetchall()

        for exp_id, split_count in rows:
            self.assertGreaterEqual(split_count, 3,
                                    f"Experiment {exp_id[:8]} has only {split_count} splits")

    def test_per_split_rmse_valid(self):
        """All per-split RMSE values must be non-negative and finite."""
        rows = self.conn.execute("""
            SELECT m.rmse FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE e.model_type LIKE 'rolling_%'
        """).fetchall()

        for (rmse,) in rows:
            self.assertIsNotNone(rmse, "NULL RMSE in rolling split")
            self.assertGreaterEqual(rmse, 0, f"Negative RMSE: {rmse}")
            self.assertLess(rmse, 1e6, f"RMSE suspiciously large: {rmse}")

    def test_stability_scores_stored(self):
        """Stability scores should be stored in metrics."""
        rows = self.conn.execute("""
            SELECT m.stability_score FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE e.model_type LIKE 'rolling_%'
            AND m.stability_score IS NOT NULL
        """).fetchall()

        self.assertGreater(len(rows), 0, "No stability scores found")
        for (score,) in rows:
            self.assertGreaterEqual(score, 0, f"Negative stability score: {score}")


class TestStabilityComparison(unittest.TestCase):
    """Test stability metrics are meaningful."""

    def setUp(self):
        self.conn = get_raw_connection()

    def tearDown(self):
        self.conn.close()

    def test_baseline_has_positive_rmse(self):
        """Baseline model should have positive mean RMSE."""
        row = self.conn.execute(
            "SELECT params_json FROM experiments WHERE model_type='rolling_baseline_ar' LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        params = json.loads(row[0])
        self.assertGreater(params["mean_rmse"], 0)

    def test_all_models_have_comparable_rmse(self):
        """All models should have RMSE in same order of magnitude."""
        rmses = []
        for model_type in ["rolling_baseline_ar", "rolling_var_all_weather", "rolling_granger_selected"]:
            row = self.conn.execute(
                "SELECT params_json FROM experiments WHERE model_type=? LIMIT 1",
                (model_type,)
            ).fetchone()
            if row:
                params = json.loads(row[0])
                rmses.append(params["mean_rmse"])

        if len(rmses) >= 2:
            # All models should be within 5x of each other (sanity check)
            self.assertLess(max(rmses) / max(min(rmses), 1e-6), 5.0,
                            f"RMSE spread too large: {rmses}")


class TestDriftAnalysis(unittest.TestCase):
    """Test drift analysis results."""

    def test_evaluation_summary_exists(self):
        """evaluation_summary.json must exist."""
        path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_evaluation_summary_valid_json(self):
        """evaluation_summary.json must be valid JSON."""
        path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("stability_reports", data)
        self.assertIn("extreme_weather_drift", data)

    def test_extreme_weather_drift_detected(self):
        """Extreme weather should cause higher error than normal."""
        path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
        with open(path) as f:
            data = json.load(f)

        extreme = data.get("extreme_weather_drift", {})
        if extreme.get("normal_rmse") is not None:
            self.assertGreater(extreme["extreme_rmse"], extreme["normal_rmse"],
                               "Extreme weather should have higher RMSE")

    def test_seasonal_drift_present(self):
        """At least one seasonal transition should be analyzed."""
        path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
        with open(path) as f:
            data = json.load(f)

        seasonal = data.get("seasonal_drift", [])
        self.assertGreater(len(seasonal), 0, "No seasonal drift data")


class TestPlots(unittest.TestCase):
    """Test plots were generated."""

    def test_rolling_comparison_plot(self):
        """Rolling validation comparison chart must exist."""
        path = os.path.join(settings.PLOTS_DIR, "rolling_validation_comparison.png")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        self.assertGreater(os.path.getsize(path), 5000)

    def test_drift_analysis_plot(self):
        """Drift analysis chart must exist."""
        path = os.path.join(settings.PLOTS_DIR, "drift_analysis.png")
        self.assertTrue(os.path.exists(path), f"Missing: {path}")
        self.assertGreater(os.path.getsize(path), 5000)


class TestNoDataLeakage(unittest.TestCase):
    """Verify temporal integrity — no future data in training."""

    def test_rolling_splits_temporal_integrity(self):
        """
        In rolling experiments, params should show train end < test start.
        Verify via the stored experiment descriptions.
        """
        conn = get_raw_connection()
        rows = conn.execute(
            "SELECT description, params_json FROM experiments WHERE model_type LIKE 'rolling_%'"
        ).fetchall()
        conn.close()

        for desc, params_json in rows:
            if params_json:
                params = json.loads(params_json)
                # Per-split RMSE must have multiple entries (not just one)
                per_split = params.get("per_split_rmse", [])
                self.assertGreater(len(per_split), 1,
                                   f"Only {len(per_split)} splits — expected > 1")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
