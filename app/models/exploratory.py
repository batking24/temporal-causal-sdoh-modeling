"""
exploratory.py — Exploratory temporal analysis.

Generates:
    1. Cross-correlation plots between weather variables and need counts
    2. Seasonal decomposition (STL-style via statsmodels)
    3. Lag-effect heatmaps showing correlation at different lag horizons

Outputs saved to outputs/plots/
"""

from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def compute_cross_correlations(
    df: pd.DataFrame,
    weather_col: str = "max_temp",
    target_col: str = "target_count",
    max_lag: int = 30,
    group_col: str = "region_id",
) -> pd.DataFrame:
    """
    Compute cross-correlation between a weather variable and target
    at lag horizons 0..max_lag, averaged across groups.

    Returns:
        DataFrame with columns: lag, mean_corr, std_corr
    """
    all_corrs = []

    for grp_name, grp_df in df.groupby(group_col):
        grp_df = grp_df.sort_values("date").reset_index(drop=True)
        if len(grp_df) < max_lag + 5:
            continue

        wx = grp_df[weather_col].values
        tgt = grp_df[target_col].values

        for lag in range(0, max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(wx, tgt)[0, 1]
            else:
                corr = np.corrcoef(wx[:-lag], tgt[lag:])[0, 1]

            if not np.isnan(corr):
                all_corrs.append({"lag": lag, "group": grp_name, "corr": corr})

    corr_df = pd.DataFrame(all_corrs)
    if corr_df.empty:
        return pd.DataFrame(columns=["lag", "mean_corr", "std_corr"])

    summary = corr_df.groupby("lag").agg(
        mean_corr=("corr", "mean"),
        std_corr=("corr", "std"),
    ).reset_index()

    return summary


def plot_cross_correlations(
    df: pd.DataFrame,
    need_type: str = "all",
    weather_vars: list[str] | None = None,
) -> str:
    """
    Generate cross-correlation plot and save to outputs/plots/.

    Returns:
        Path to saved plot.
    """
    if weather_vars is None:
        weather_vars = ["max_temp", "precip"]

    fig, axes = plt.subplots(1, len(weather_vars), figsize=(6 * len(weather_vars), 4))
    if len(weather_vars) == 1:
        axes = [axes]

    for ax, wx_col in zip(axes, weather_vars):
        if wx_col not in df.columns:
            continue
        cc = compute_cross_correlations(df, weather_col=wx_col, target_col="target_count")
        if cc.empty:
            continue

        ax.bar(cc["lag"], cc["mean_corr"], alpha=0.7, color="#3b82f6")
        ax.fill_between(
            cc["lag"],
            cc["mean_corr"] - cc["std_corr"],
            cc["mean_corr"] + cc["std_corr"],
            alpha=0.2, color="#3b82f6",
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Cross-correlation")
        ax.set_title(f"{wx_col} → target ({need_type})")

    plt.tight_layout()
    plot_path = os.path.join(settings.PLOTS_DIR, f"cross_corr_{need_type}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved cross-correlation plot: %s", plot_path)
    return plot_path


def plot_lag_heatmap(
    granger_results: pd.DataFrame,
) -> str:
    """
    Generate a lag-significance heatmap from Granger test results.

    Args:
        granger_results: DataFrame with columns: feature_name, lag, p_value, need_type

    Returns:
        Path to saved heatmap.
    """
    if granger_results.empty:
        logger.warning("No Granger results to plot")
        return ""

    # Pivot: features × lags → -log10(p_value)
    plot_data = granger_results.copy()
    plot_data["neg_log_p"] = -np.log10(plot_data["p_value"].clip(lower=1e-10))

    pivot = plot_data.pivot_table(
        index="feature_name",
        columns="lag",
        values="neg_log_p",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Weather Feature")
    ax.set_title("Granger Causality Significance: -log₁₀(p-value)")

    # Significance threshold line
    plt.colorbar(im, ax=ax, label="-log₁₀(p)")

    plt.tight_layout()
    plot_path = os.path.join(settings.PLOTS_DIR, "lag_significance_heatmap.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved lag significance heatmap: %s", plot_path)
    return plot_path
