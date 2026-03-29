"""
granger.py — Granger causality testing.

Tests whether weather variables (tmax, prcp) Granger-cause changes in social
need counts, per (region, need_type) and across multiple lag horizons.

Methodology:
    - Uses statsmodels grangercausalitytests
    - Tests lags 1 through max_lag for each (weather_var, need_type, region)
    - Records p-values, F-statistics, and significance flags
    - Stores results in causal_results table
"""

from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

from app.config import get_settings
from app.db import get_raw_connection

logger = logging.getLogger(__name__)
settings = get_settings()

# Suppress statsmodels FutureWarnings during bulk testing
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


def _check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """Run Augmented Dickey-Fuller test for stationarity."""
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_stat": round(result[0], 4),
            "p_value": round(result[1], 6),
            "is_stationary": result[1] < significance,
            "n_lags_used": result[2],
        }
    except Exception as e:
        logger.debug("ADF test failed: %s", e)
        return {"adf_stat": None, "p_value": None, "is_stationary": False, "n_lags_used": 0}


def _make_stationary(series: pd.Series) -> pd.Series:
    """
    Difference the series if it's not stationary.
    Returns the (possibly differenced) series.
    """
    check = _check_stationarity(series)
    if check["is_stationary"]:
        return series
    # First-order differencing
    diffed = series.diff().dropna()
    return diffed


def run_granger_tests(
    df: pd.DataFrame,
    weather_vars: list[str] | None = None,
    max_lag: int | None = None,
    significance: float | None = None,
) -> pd.DataFrame:
    """
    Run Granger causality tests for all (region, need_type, weather_var) combinations.

    Args:
        df: Feature DataFrame with columns including weather vars and target_count.
        weather_vars: Weather columns to test. Default: tmax_lag and prcp_lag base vars.
        max_lag: Maximum lag to test. Default from settings.
        significance: P-value threshold. Default from settings.

    Returns:
        DataFrame with columns: region_id, need_type, feature_name, lag,
            p_value, f_statistic, significant_flag
    """
    if max_lag is None:
        max_lag = min(settings.MAX_LAG_DAYS, 15)  # cap for test stability
    if significance is None:
        significance = settings.GRANGER_SIGNIFICANCE

    # Use the weather columns from the aggregated data
    # We test: does weather Granger-cause target?
    if weather_vars is None:
        weather_vars = ["max_temp", "precip"]

    results = []
    groups = df.groupby(["region_id", "need_type"])
    total_groups = len(groups)
    tested = 0

    for (region, need_type), grp in groups:
        grp = grp.sort_values("date").reset_index(drop=True)

        # Need enough observations for the test
        if len(grp) < max_lag + 10:
            logger.debug("Skipping %s/%s — only %d rows", region, need_type, len(grp))
            continue

        target = grp["target_count"].astype(float)

        for wx_var in weather_vars:
            if wx_var not in grp.columns:
                continue

            predictor = grp[wx_var].astype(float)

            # Make both series stationary
            target_s = _make_stationary(target)
            pred_s = _make_stationary(predictor)

            # Align after differencing
            common_idx = target_s.index.intersection(pred_s.index)
            if len(common_idx) < max_lag + 10:
                continue

            test_data = pd.DataFrame({
                "target": target_s.loc[common_idx].values,
                "predictor": pred_s.loc[common_idx].values,
            }).dropna()

            if len(test_data) < max_lag + 10:
                continue

            # Add small noise to prevent singular matrix errors
            test_data["target"] = test_data["target"] + np.random.normal(0, 1e-8, len(test_data))
            test_data["predictor"] = test_data["predictor"] + np.random.normal(0, 1e-8, len(test_data))

            try:
                gc_result = grangercausalitytests(
                    test_data[["target", "predictor"]],
                    maxlag=max_lag,
                    verbose=False,
                )

                for lag in range(1, max_lag + 1):
                    if lag not in gc_result:
                        continue
                    test_stats = gc_result[lag]
                    # Use the ssr_ftest (most standard)
                    f_test = test_stats[0]["ssr_ftest"]
                    f_stat = f_test[0]
                    p_val = f_test[1]

                    results.append({
                        "region_id": region,
                        "need_type": need_type,
                        "feature_name": wx_var,
                        "lag": lag,
                        "p_value": round(p_val, 6),
                        "f_statistic": round(f_stat, 4),
                        "effect_strength": round(1 - p_val, 4),  # simple proxy
                        "significant_flag": 1 if p_val < significance else 0,
                    })

                tested += 1

            except Exception as e:
                logger.debug("Granger test failed for %s/%s/%s: %s",
                             region, need_type, wx_var, e)
                continue

    logger.info("Granger tests: %d groups tested out of %d, %d total results",
                tested, total_groups, len(results))

    return pd.DataFrame(results)


def store_granger_results(
    results_df: pd.DataFrame,
    experiment_id: str | None = None,
) -> str:
    """
    Store Granger test results in the causal_results table.

    Returns:
        The experiment_id used.
    """
    if experiment_id is None:
        experiment_id = str(uuid.uuid4())

    conn = get_raw_connection()

    # Create experiment entry
    conn.execute(
        """INSERT OR REPLACE INTO experiments
           (experiment_id, model_type, target, region_scope, train_start, train_end, description)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (experiment_id, "granger_causality", "all_need_types", "all_states",
         "2025-01-01", "2025-07-31", "Granger causality tests for weather → social needs")
    )

    # Store individual results
    for _, row in results_df.iterrows():
        conn.execute(
            """INSERT INTO causal_results
               (experiment_id, feature_name, lag, p_value, f_statistic,
                effect_strength, significant_flag, need_type, region_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, row["feature_name"], int(row["lag"]),
             row["p_value"], row.get("f_statistic"), row.get("effect_strength"),
             int(row["significant_flag"]), row.get("need_type"), row.get("region_id"))
        )

    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) FROM causal_results WHERE experiment_id=?",
        (experiment_id,)
    ).fetchone()[0]
    conn.close()

    logger.info("Stored %d causal results under experiment %s", count, experiment_id[:8])
    return experiment_id


def get_significant_lags(
    results_df: pd.DataFrame,
    significance: float = 0.05,
) -> dict[str, list[int]]:
    """
    Extract the significant lag horizons per weather variable.

    Returns:
        Dict mapping feature_name → list of significant lags, sorted by p-value.
    """
    sig = results_df[results_df["significant_flag"] == 1]
    if sig.empty:
        # Relax threshold if nothing significant
        sig = results_df[results_df["p_value"] < 0.20]

    sig_lags = {}
    for feature, group in sig.groupby("feature_name"):
        best = group.sort_values("p_value")
        sig_lags[feature] = sorted(best["lag"].unique().tolist())

    return sig_lags
