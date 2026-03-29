"""
train_eval.py — Orchestrator for all modeling in Phase 5.

Pipeline:
    1. Load feature matrix from model_features_daily
    2. Run Granger causality tests → identify significant lags
    3. Run exploratory analysis → generate plots
    4. Simple train/test split (last month = test)
    5. Train all 3 models:
        a. Baseline AR (target-only)
        b. VAR all-weather (all weather features)
        c. Granger-selected (only significant lags)
    6. Store all results in experiments + metrics tables
    7. Print comparison summary
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import get_raw_connection
from app.models.granger import run_granger_tests, store_granger_results, get_significant_lags
from app.models.baseline_ar import train_baseline_ar
from app.models.var_model import train_weather_model, train_granger_selected_model
from app.models.exploratory import plot_cross_correlations, plot_lag_heatmap

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_features() -> pd.DataFrame:
    """Load the full feature matrix."""
    conn = get_raw_connection()
    df = pd.read_sql("SELECT * FROM model_features_daily", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %d feature rows", len(df))
    return df


def _create_train_test_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Create temporal train/test split.
    Test = last month of data; Train = everything before.
    """
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=30)

    train_mask = df["date"] < test_start
    test_mask = df["date"] >= test_start

    logger.info("Train: %d rows (< %s), Test: %d rows (>= %s)",
                train_mask.sum(), test_start.date(), test_mask.sum(), test_start.date())
    return train_mask, test_mask


def _store_experiment_and_metrics(
    model_result: dict,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> str:
    """Store an experiment and its metrics in the database."""
    experiment_id = str(uuid.uuid4())
    conn = get_raw_connection()

    conn.execute(
        """INSERT INTO experiments
           (experiment_id, model_type, target, region_scope,
            train_start, train_end, test_start, test_end, params_json, description)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (experiment_id, model_result["model_type"], "all_need_types", "all_states",
         train_start, train_end, test_start, test_end,
         json.dumps(model_result.get("feature_importance", {})),
         f"{model_result['model_type']} model — Phase 5 initial run")
    )

    conn.execute(
        """INSERT INTO metrics
           (experiment_id, split_index, rmse, mae, mape)
           VALUES (?, ?, ?, ?, ?)""",
        (experiment_id, 0, model_result["rmse"], model_result["mae"],
         model_result.get("mape"))
    )

    conn.commit()
    conn.close()
    return experiment_id


def run_full_modeling_pipeline() -> dict:
    """
    Execute the complete Phase 5 modeling pipeline.

    Returns:
        Summary dict with results from all models.
    """
    start = time.time()
    results = {}

    # 1. Load features
    df = _load_features()

    # 2. Granger causality
    logger.info("=" * 60)
    logger.info("STEP 1: Granger Causality Tests")
    logger.info("=" * 60)

    # We need to join weather agg data for cross-corr and Granger
    conn = get_raw_connection()
    weather_agg = pd.read_sql("SELECT * FROM weather_daily_agg", conn)
    conn.close()
    weather_agg["date"] = pd.to_datetime(weather_agg["date"])

    # Merge weather into feature df for Granger testing
    df_with_wx = df.merge(
        weather_agg[["date", "region_id", "max_temp", "precip"]],
        on=["date", "region_id"],
        how="left",
    )

    granger_df = run_granger_tests(df_with_wx, weather_vars=["max_temp", "precip"], max_lag=15)
    granger_exp_id = store_granger_results(granger_df)
    significant_lags = get_significant_lags(granger_df)

    results["granger"] = {
        "total_tests": len(granger_df),
        "significant_tests": int((granger_df["significant_flag"] == 1).sum()) if not granger_df.empty else 0,
        "significant_lags": significant_lags,
        "experiment_id": granger_exp_id,
    }

    # 3. Exploratory plots
    logger.info("=" * 60)
    logger.info("STEP 2: Exploratory Analysis & Plots")
    logger.info("=" * 60)

    try:
        plot_cross_correlations(df_with_wx, need_type="all", weather_vars=["max_temp", "precip"])
        if not granger_df.empty:
            plot_lag_heatmap(granger_df)
    except Exception as e:
        logger.warning("Plot generation failed (non-critical): %s", e)

    # 4. Train/test split
    logger.info("=" * 60)
    logger.info("STEP 3: Model Training & Evaluation")
    logger.info("=" * 60)

    train_mask, test_mask = _create_train_test_split(df)

    train_dates = df[train_mask]["date"]
    test_dates = df[test_mask]["date"]
    train_start = str(train_dates.min().date())
    train_end = str(train_dates.max().date())
    test_start = str(test_dates.min().date()) if test_mask.sum() > 0 else train_end
    test_end = str(test_dates.max().date()) if test_mask.sum() > 0 else train_end

    # 5a. Baseline AR
    logger.info("Training baseline AR model...")
    baseline = train_baseline_ar(df, train_mask, test_mask)
    baseline_id = _store_experiment_and_metrics(
        baseline, train_start, train_end, test_start, test_end
    )
    results["baseline_ar"] = baseline

    # 5b. VAR all-weather
    logger.info("Training VAR all-weather model...")
    var_all = train_weather_model(df, train_mask, test_mask, model_name="var_all_weather")
    var_all_id = _store_experiment_and_metrics(
        var_all, train_start, train_end, test_start, test_end
    )
    results["var_all_weather"] = var_all

    # 5c. Granger-selected
    logger.info("Training Granger-selected model...")
    granger_model = train_granger_selected_model(
        df, train_mask, test_mask, significant_lags
    )
    granger_model_id = _store_experiment_and_metrics(
        granger_model, train_start, train_end, test_start, test_end
    )
    results["granger_selected"] = granger_model

    elapsed = round(time.time() - start, 2)
    results["elapsed_sec"] = elapsed

    return results


def print_comparison(results: dict) -> None:
    """Pretty-print model comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    models = ["baseline_ar", "var_all_weather", "granger_selected"]
    headers = ["Model", "RMSE", "MAE", "MAPE%", "Train", "Test", "Features"]
    print(f"  {headers[0]:22s} {headers[1]:>8s} {headers[2]:>8s} {headers[3]:>8s} "
          f"{headers[4]:>6s} {headers[5]:>5s} {headers[6]:>8s}")
    print("  " + "-" * 68)

    for model_name in models:
        m = results.get(model_name, {})
        if not m:
            continue
        print(f"  {model_name:22s} {m.get('rmse', 'N/A'):>8} {m.get('mae', 'N/A'):>8} "
              f"{m.get('mape', 'N/A'):>8} {m.get('n_train', '-'):>6} "
              f"{m.get('n_test', '-'):>5} {m.get('n_features', '-'):>8}")

    # Improvement calculation
    baseline_rmse = results.get("baseline_ar", {}).get("rmse")
    granger_rmse = results.get("granger_selected", {}).get("rmse")
    var_rmse = results.get("var_all_weather", {}).get("rmse")

    if baseline_rmse and granger_rmse and not np.isnan(baseline_rmse) and not np.isnan(granger_rmse):
        improvement = (baseline_rmse - granger_rmse) / baseline_rmse * 100
        print(f"\n  RMSE improvement (Granger-selected vs baseline): {improvement:+.1f}%")

    if baseline_rmse and var_rmse and not np.isnan(baseline_rmse) and not np.isnan(var_rmse):
        improvement = (baseline_rmse - var_rmse) / baseline_rmse * 100
        print(f"  RMSE improvement (all-weather vs baseline):      {improvement:+.1f}%")

    # Granger summary
    granger = results.get("granger", {})
    print(f"\n  Granger causality: {granger.get('significant_tests', 0)}/{granger.get('total_tests', 0)} "
          f"significant tests")
    if granger.get("significant_lags"):
        for var, lags in granger["significant_lags"].items():
            print(f"    {var}: lags {lags}")

    print(f"\n  Total elapsed: {results.get('elapsed_sec', '?')}s")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    results = run_full_modeling_pipeline()
    print_comparison(results)
    print("\n✅ Phase 5 modeling complete.")
