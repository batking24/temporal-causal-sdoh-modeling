"""
metrics.py — Forecasting and stability metric computations.

Metrics:
    Forecasting:
        - RMSE, MAE, MAPE, normalized RMSE (NRMSE)
    Stability:
        - stability_score = σ(RMSE) across rolling splits
        - error_variance = variance of absolute errors
    Comparison:
        - Δ RMSE, Δ stability between models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    """Container for a single evaluation split's metrics."""
    rmse: float
    mae: float
    mape: float
    nrmse: float          # normalized RMSE = RMSE / mean(actual)
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityReport:
    """Aggregated stability metrics across rolling splits."""
    model_type: str
    mean_rmse: float
    std_rmse: float       # = stability_score (lower = more stable)
    mean_mae: float
    std_mae: float
    mean_mape: float
    n_splits: int
    per_split_rmse: list[float]

    @property
    def stability_score(self) -> float:
        """Lower is better — std dev of RMSE across temporal splits."""
        return self.std_rmse

    @property
    def cv_rmse(self) -> float:
        """Coefficient of variation of RMSE (relative stability)."""
        if self.mean_rmse == 0:
            return 0.0
        return self.std_rmse / self.mean_rmse

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stability_score"] = self.stability_score
        d["cv_rmse"] = round(self.cv_rmse, 4)
        return d


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ForecastMetrics:
    """Compute forecasting metrics for a single split."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return ForecastMetrics(rmse=np.nan, mae=np.nan, mape=np.nan, nrmse=np.nan, n_samples=0)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(abs_errors))

    # MAPE — avoid division by zero
    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100)
    else:
        mape = 0.0

    # Normalized RMSE
    mean_actual = float(np.mean(y_true))
    nrmse = rmse / mean_actual if mean_actual != 0 else 0.0

    return ForecastMetrics(
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        mape=round(mape, 2),
        nrmse=round(nrmse, 4),
        n_samples=n,
    )


def compute_stability_report(
    model_type: str,
    per_split_metrics: list[ForecastMetrics],
) -> StabilityReport:
    """
    Aggregate per-split metrics into a stability report.

    Args:
        model_type: Name of the model.
        per_split_metrics: List of ForecastMetrics, one per rolling split.

    Returns:
        StabilityReport with mean, std, and per-split breakdowns.
    """
    if not per_split_metrics:
        return StabilityReport(
            model_type=model_type, mean_rmse=np.nan, std_rmse=np.nan,
            mean_mae=np.nan, std_mae=np.nan, mean_mape=np.nan,
            n_splits=0, per_split_rmse=[],
        )

    rmses = [m.rmse for m in per_split_metrics if not np.isnan(m.rmse)]
    maes = [m.mae for m in per_split_metrics if not np.isnan(m.mae)]
    mapes = [m.mape for m in per_split_metrics if not np.isnan(m.mape)]

    return StabilityReport(
        model_type=model_type,
        mean_rmse=round(float(np.mean(rmses)), 4) if rmses else np.nan,
        std_rmse=round(float(np.std(rmses)), 4) if rmses else np.nan,
        mean_mae=round(float(np.mean(maes)), 4) if maes else np.nan,
        std_mae=round(float(np.std(maes)), 4) if maes else np.nan,
        mean_mape=round(float(np.mean(mapes)), 2) if mapes else np.nan,
        n_splits=len(rmses),
        per_split_rmse=[round(r, 4) for r in rmses],
    )
