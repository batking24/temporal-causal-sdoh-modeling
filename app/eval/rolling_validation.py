"""
rolling_validation.py — Time-split rolling (expanding window) validation.

Strategy:
    For each split i:
        Train on months [1..N-K+i]
        Test on month [N-K+i+1]
    where K = number of splits.

    This ensures:
        - No data leakage (test always after train)
        - Models are retrained per split
        - Metrics are collected per split for stability analysis
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import get_raw_connection
from app.eval.metrics import compute_forecast_metrics, compute_stability_report, ForecastMetrics, StabilityReport
from app.models.baseline_ar import train_baseline_ar
from app.models.var_model import train_weather_model, train_granger_selected_model
from app.models.granger import get_significant_lags

logger = logging.getLogger(__name__)
settings = get_settings()


def _generate_temporal_splits(
    df: pd.DataFrame,
    min_train_days: int = 60,
    test_window_days: int = 14,
) -> list[tuple[pd.Series, pd.Series, str, str]]:
    """
    Generate expanding-window temporal splits.

    Each split:
        - Train: all data from start up to split point
        - Test: next test_window_days after split point

    Returns:
        List of (train_mask, test_mask, test_start_str, test_end_str)
    """
    df_sorted = df.sort_values("date")
    min_date = df_sorted["date"].min()
    max_date = df_sorted["date"].max()
    total_days = (max_date - min_date).days

    splits = []
    split_start = min_date + pd.Timedelta(days=min_train_days)

    while split_start + pd.Timedelta(days=test_window_days) <= max_date:
        test_end = split_start + pd.Timedelta(days=test_window_days)

        train_mask = df["date"] < split_start
        test_mask = (df["date"] >= split_start) & (df["date"] < test_end)

        if train_mask.sum() >= 20 and test_mask.sum() >= 5:
            splits.append((
                train_mask,
                test_mask,
                str(split_start.date()),
                str(test_end.date()),
            ))

        split_start += pd.Timedelta(days=test_window_days)

    logger.info("Generated %d temporal splits (train min %d days, test window %d days)",
                len(splits), min_train_days, test_window_days)
    return splits


def run_rolling_validation(
    df: pd.DataFrame,
    significant_lags: dict[str, list[int]] | None = None,
) -> dict[str, StabilityReport]:
    """
    Run rolling temporal validation for all three model types.

    Args:
        df: Full feature DataFrame.
        significant_lags: Granger-significant lags for the selected model.

    Returns:
        Dict mapping model_type → StabilityReport.
    """
    start = time.time()

    if significant_lags is None:
        significant_lags = {}

    splits = _generate_temporal_splits(df, min_train_days=45, test_window_days=14)

    if not splits:
        logger.warning("No valid temporal splits generated")
        return {}

    model_types = ["baseline_ar", "var_all_weather", "granger_selected"]
    per_model_metrics: dict[str, list[ForecastMetrics]] = {m: [] for m in model_types}

    for split_idx, (train_mask, test_mask, test_start, test_end) in enumerate(splits):
        logger.info("Split %d/%d: test [%s, %s) — train=%d, test=%d",
                     split_idx + 1, len(splits), test_start, test_end,
                     train_mask.sum(), test_mask.sum())

        # Baseline AR
        result = train_baseline_ar(df, train_mask, test_mask)
        if result["predictions"]:
            m = compute_forecast_metrics(
                np.array(result["actuals"]), np.array(result["predictions"])
            )
            per_model_metrics["baseline_ar"].append(m)

        # VAR all-weather
        result = train_weather_model(df, train_mask, test_mask, model_name="var_all_weather")
        if result["predictions"]:
            m = compute_forecast_metrics(
                np.array(result["actuals"]), np.array(result["predictions"])
            )
            per_model_metrics["var_all_weather"].append(m)

        # Granger-selected
        result = train_granger_selected_model(df, train_mask, test_mask, significant_lags)
        if result["predictions"]:
            m = compute_forecast_metrics(
                np.array(result["actuals"]), np.array(result["predictions"])
            )
            per_model_metrics["granger_selected"].append(m)

    # Build stability reports
    reports = {}
    for model_type in model_types:
        reports[model_type] = compute_stability_report(
            model_type, per_model_metrics[model_type]
        )

    elapsed = round(time.time() - start, 2)
    logger.info("Rolling validation complete: %d splits, %.2fs", len(splits), elapsed)

    return reports


def store_rolling_results(
    reports: dict[str, StabilityReport],
) -> dict[str, str]:
    """
    Store rolling validation results in experiments + metrics tables.

    Returns:
        Dict mapping model_type → experiment_id.
    """
    conn = get_raw_connection()
    experiment_ids = {}

    for model_type, report in reports.items():
        exp_id = str(uuid.uuid4())
        experiment_ids[model_type] = exp_id

        conn.execute(
            """INSERT INTO experiments
               (experiment_id, model_type, target, region_scope,
                train_start, train_end, params_json, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, f"rolling_{model_type}", "all_need_types", "all_states",
             "2025-01-01", "2025-07-31",
             json.dumps(report.to_dict()),
             f"Rolling validation — {model_type} ({report.n_splits} splits)")
        )

        # Store per-split metrics
        for split_idx, rmse_val in enumerate(report.per_split_rmse):
            metric = report  # use report-level aggregates + per-split RMSE
            conn.execute(
                """INSERT INTO metrics
                   (experiment_id, split_index, rmse, mae, mape, stability_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (exp_id, split_idx, rmse_val, report.mean_mae,
                 report.mean_mape, report.stability_score)
            )

    conn.commit()
    conn.close()

    logger.info("Stored rolling results for %d model types", len(experiment_ids))
    return experiment_ids
