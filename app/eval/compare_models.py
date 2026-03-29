"""
compare_models.py — Orchestrator for Phase 6: rolling validation + drift + comparison.

Pipeline:
    1. Load features + Granger significant lags
    2. Run rolling temporal validation for all 3 models
    3. Store per-split metrics in DB
    4. Run drift analysis (seasonal + extreme weather)
    5. Generate comparison charts and summary table
    6. Compute % improvement metrics for resume claims
"""

from __future__ import annotations

import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import get_raw_connection
from app.eval.rolling_validation import run_rolling_validation, store_rolling_results
from app.eval.drift_analysis import analyze_seasonal_drift, analyze_extreme_weather_performance
from app.eval.metrics import StabilityReport
from app.models.var_model import ALL_WEATHER_FEATURES, BASE_FEATURES

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_features() -> pd.DataFrame:
    """Load the full feature matrix."""
    conn = get_raw_connection()
    df = pd.read_sql("SELECT * FROM model_features_daily", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_significant_lags() -> dict[str, list[int]]:
    """Load Granger-significant lags from the DB."""
    conn = get_raw_connection()
    rows = conn.execute(
        "SELECT DISTINCT feature_name, lag FROM causal_results "
        "WHERE significant_flag = 1 ORDER BY feature_name, lag"
    ).fetchall()
    conn.close()

    lags: dict[str, list[int]] = {}
    for feature, lag in rows:
        lags.setdefault(feature, []).append(lag)

    if not lags:
        # Fallback: use top lags by p-value
        conn = get_raw_connection()
        rows = conn.execute(
            "SELECT feature_name, lag FROM causal_results "
            "WHERE p_value < 0.20 ORDER BY p_value LIMIT 20"
        ).fetchall()
        conn.close()
        for feature, lag in rows:
            lags.setdefault(feature, []).append(lag)

    return lags


def _plot_rolling_comparison(reports: dict[str, StabilityReport]) -> str:
    """Generate split-wise RMSE comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-split RMSE
    ax = axes[0]
    models = list(reports.keys())
    max_splits = max(r.n_splits for r in reports.values())

    x = np.arange(max_splits)
    width = 0.25
    colors = ["#6366f1", "#10b981", "#f59e0b"]

    for i, (model, report) in enumerate(reports.items()):
        rmses = report.per_split_rmse + [0] * (max_splits - len(report.per_split_rmse))
        ax.bar(x + i * width, rmses[:max_splits], width, label=model, alpha=0.85, color=colors[i])

    ax.set_xlabel("Split Index")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-Split RMSE — Rolling Validation")
    ax.legend(fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"S{i+1}" for i in range(max_splits)])

    # Summary comparison
    ax = axes[1]
    model_names = list(reports.keys())
    mean_rmses = [reports[m].mean_rmse for m in model_names]
    std_rmses = [reports[m].std_rmse for m in model_names]

    bars = ax.bar(model_names, mean_rmses, yerr=std_rmses, capsize=5,
                  color=colors[:len(model_names)], alpha=0.85)
    ax.set_ylabel("Mean RMSE (± σ)")
    ax.set_title("Model Stability Comparison")

    for bar, val, std in zip(bars, mean_rmses, std_rmses):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f"{val:.1f}±{std:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(settings.PLOTS_DIR, "rolling_validation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved rolling comparison plot: %s", path)
    return path


def _plot_drift_analysis(seasonal_df: pd.DataFrame, extreme_results: dict) -> str:
    """Generate drift analysis visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Seasonal drift
    ax = axes[0]
    if not seasonal_df.empty:
        labels = [f"{r['train_season']} to {r['test_season']}" for _, r in seasonal_df.iterrows()]
        rmses = seasonal_df["rmse"].values
        ax.bar(labels, rmses, color=["#3b82f6", "#ef4444"], alpha=0.85)
        ax.set_ylabel("RMSE")
        ax.set_title("Seasonal Transition Error")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Seasonal Transition Error")

    # Extreme weather drift Comparison
    ax = axes[1]
    granger = extreme_results.get("granger_selected", {})
    baseline = extreme_results.get("baseline", {})

    if granger.get("normal_rmse") is not None and baseline.get("normal_rmse") is not None:
        labels = ["Normal", "Extreme"]
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, [baseline["normal_rmse"], baseline["extreme_rmse"]], 
               width, label="Baseline", color="#6366f1", alpha=0.7)
        ax.bar(x + width/2, [granger["normal_rmse"], granger["extreme_rmse"]], 
               width, label="Granger-Selected", color="#f59e0b", alpha=0.9)
        
        ax.set_ylabel("RMSE")
        ax.set_title("Robustness Gap: Baseline vs. Proposed")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Weather Regime Performance")

    plt.tight_layout()
    path = os.path.join(settings.PLOTS_DIR, "drift_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved drift analysis plot: %s", path)
    return path


def run_full_evaluation() -> dict:
    """
    Execute the complete Phase 6 evaluation pipeline.

    Returns:
        Summary dict with stability reports, drift analysis, and improvement metrics.
    """
    start = time.time()
    results = {}

    # 1. Load data
    df = _load_features()
    sig_lags = _load_significant_lags()
    logger.info("Significant lags: %s", sig_lags)

    # 2. Rolling validation
    logger.info("=" * 60)
    logger.info("STEP 1: Rolling Temporal Validation")
    logger.info("=" * 60)

    reports = run_rolling_validation(df, sig_lags)
    exp_ids = store_rolling_results(reports)

    results["stability_reports"] = {k: v.to_dict() for k, v in reports.items()}

    # 3. Drift analysis
    logger.info("=" * 60)
    logger.info("STEP 2: Drift Analysis")
    logger.info("=" * 60)

    # Calculate drift for Granger-selected model (Full)
    model_features = BASE_FEATURES + ALL_WEATHER_FEATURES
    seasonal_df = analyze_seasonal_drift(df, model_features)
    extreme_granger = analyze_extreme_weather_performance(df, model_features)

    # Calculate drift for Baseline model (Lag-only)
    extreme_baseline = analyze_extreme_weather_performance(df, BASE_FEATURES)

    results["seasonal_drift"] = seasonal_df.to_dict("records") if not seasonal_df.empty else []
    results["extreme_weather_drift"] = {
        "granger_selected": extreme_granger,
        "baseline": extreme_baseline
    }

    # 4. Generate plots
    logger.info("=" * 60)
    logger.info("STEP 3: Generating Comparison Charts")
    logger.info("=" * 60)

    try:
        _plot_rolling_comparison(reports)
        _plot_drift_analysis(seasonal_df, results["extreme_weather_drift"])
    except Exception as e:
        logger.warning("Plot generation failed: %s", e)

    # 5. Compute improvement metrics
    baseline_report = reports.get("baseline_ar")
    granger_report = reports.get("granger_selected")
    var_report = reports.get("var_all_weather")

    improvements = {}
    if baseline_report and granger_report:
        if not np.isnan(baseline_report.mean_rmse) and not np.isnan(granger_report.mean_rmse):
            rmse_imp = (baseline_report.mean_rmse - granger_report.mean_rmse) / baseline_report.mean_rmse * 100
            improvements["granger_vs_baseline_rmse_pct"] = round(rmse_imp, 1)

        if not np.isnan(baseline_report.stability_score) and not np.isnan(granger_report.stability_score):
            stab_imp = (baseline_report.stability_score - granger_report.stability_score) / max(baseline_report.stability_score, 1e-6) * 100
            improvements["granger_vs_baseline_stability_pct"] = round(stab_imp, 1)

    if baseline_report and var_report:
        if not np.isnan(baseline_report.mean_rmse) and not np.isnan(var_report.mean_rmse):
            rmse_imp = (baseline_report.mean_rmse - var_report.mean_rmse) / baseline_report.mean_rmse * 100
            improvements["var_vs_baseline_rmse_pct"] = round(rmse_imp, 1)

    if extreme_granger and extreme_baseline:
        if extreme_granger.get("extreme_rmse") and extreme_baseline.get("extreme_rmse"):
            robustness_imp = (extreme_baseline["extreme_rmse"] - extreme_granger["extreme_rmse"]) / extreme_baseline["extreme_rmse"] * 100
            improvements["robustness_gain_pct"] = round(robustness_imp, 1)

    results["improvements"] = improvements

    elapsed = round(time.time() - start, 2)
    results["elapsed_sec"] = elapsed

    return results


def print_evaluation_summary(results: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("PHASE 6 — ROLLING VALIDATION & STABILITY ANALYSIS")
    print("=" * 70)

    # Stability comparison table
    print("\n  MODEL STABILITY COMPARISON (Rolling Temporal Splits)")
    print("  " + "-" * 66)
    print(f"  {'Model':22s} {'Mean RMSE':>10s} {'σ(RMSE)':>10s} {'Mean MAE':>10s} "
          f"{'Splits':>7s} {'CV':>7s}")
    print("  " + "-" * 66)

    for model_type, report in results.get("stability_reports", {}).items():
        print(f"  {model_type:22s} {report['mean_rmse']:>10.2f} {report['std_rmse']:>10.2f} "
              f"{report['mean_mae']:>10.2f} {report['n_splits']:>7d} {report.get('cv_rmse', 0):>7.3f}")

    # Improvements
    imp = results.get("improvements", {})
    if imp:
        print(f"\n  IMPROVEMENT METRICS")
        print("  " + "-" * 50)
        for key, val in imp.items():
            direction = "better" if val > 0 else "worse"
            print(f"  {key:45s} {val:>+6.1f}%  ({direction})")

    # Drift analysis
    extreme = results.get("extreme_weather_drift", {})
    if extreme.get("granger_selected") and extreme.get("baseline"):
        g = extreme["granger_selected"]
        b = extreme["baseline"]
        print(f"\n  DISTRIBUTION SHIFT ROBUSTNESS (Extreme Weather)")
        print("  " + "-" * 50)
        print(f"  {'Metric':25s} {'Baseline':>10s} {'Proposed':>10s}")
        print("  " + "-" * 50)
        print(f"  {'Normal Weather RMSE':25s} {b.get('normal_rmse', 0):>10.2f} {g.get('normal_rmse', 0):>10.2f}")
        print(f"  {'Extreme Weather RMSE':25s} {b.get('extreme_rmse', 0):>10.2f} {g.get('extreme_rmse', 0):>10.2f}")
        print(f"  {'Relative Drift (%)':25s} {b.get('relative_increase_pct', 0):>10.1f}% {g.get('relative_increase_pct', 0):>10.1f}%")

    # Seasonal drift
    seasonal = results.get("seasonal_drift", [])
    if seasonal:
        print(f"\n  SEASONAL DRIFT")
        print("  " + "-" * 50)
        for s in seasonal:
            print(f"  {s['train_season']:8s} to {s['test_season']:8s}  RMSE={s['rmse']:.2f}  "
                  f"(train={s['n_train']}, test={s['n_test']})")

    print(f"\n  Total elapsed: {results.get('elapsed_sec', '?')}s")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    results = run_full_evaluation()
    print_evaluation_summary(results)

    # Save summary to JSON
    summary_path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved evaluation summary: %s", summary_path)

    print("\nPhase 6 evaluation complete.")
