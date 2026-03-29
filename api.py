"""
api.py — FastAPI REST API for programmatic access to causal analysis results.

Endpoints:
    GET  /health                → pipeline health check
    GET  /api/v1/summary        → executive summary metrics
    GET  /api/v1/granger        → Granger causality results with filters
    GET  /api/v1/models         → model comparison table
    GET  /api/v1/drift          → drift analysis results
    GET  /api/v1/features/{region}/{need_type}  → feature time series
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import get_raw_connection

settings = get_settings()

app = FastAPI(
    title="Climate → Social Needs Causal Analysis API",
    description="Programmatic access to temporal causal modeling results",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    """Pipeline health check."""
    conn = get_raw_connection()
    tables = {}
    for t in ["raw_social_needs", "raw_weather_daily", "region_lookup",
              "weather_daily_agg", "social_needs_daily_agg", "model_features_daily",
              "experiments", "metrics", "causal_results"]:
        tables[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    conn.close()
    return {"status": "healthy", "tables": tables}


@app.get("/api/v1/summary")
def summary():
    """Executive summary metrics."""
    conn = get_raw_connection()
    result = {
        "raw_records": conn.execute("SELECT COUNT(*) FROM raw_social_needs").fetchone()[0],
        "states": conn.execute("SELECT COUNT(DISTINCT region_id) FROM weather_daily_agg").fetchone()[0],
        "need_categories": conn.execute("SELECT COUNT(DISTINCT need_type) FROM social_needs_daily_agg").fetchone()[0],
        "feature_rows": conn.execute("SELECT COUNT(*) FROM model_features_daily").fetchone()[0],
        "granger_tests": conn.execute("SELECT COUNT(*) FROM causal_results").fetchone()[0],
        "significant_tests": conn.execute("SELECT COUNT(*) FROM causal_results WHERE significant_flag=1").fetchone()[0],
        "experiments": conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
    }
    conn.close()

    # Load eval summary if available
    summary_path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            eval_data = json.load(f)
        result["extreme_weather_drift"] = eval_data.get("extreme_weather_drift", {})
        result["stability_reports"] = eval_data.get("stability_reports", {})

    return result


@app.get("/api/v1/granger")
def granger_results(
    feature: Optional[str] = Query(None, description="Filter by weather feature"),
    significant_only: bool = Query(False, description="Only significant results"),
    max_lag: Optional[int] = Query(None, description="Max lag to return"),
):
    """Granger causality test results."""
    conn = get_raw_connection()
    query = "SELECT * FROM causal_results WHERE 1=1"
    params = []

    if feature:
        query += " AND feature_name = ?"
        params.append(feature)
    if significant_only:
        query += " AND significant_flag = 1"
    if max_lag:
        query += " AND lag <= ?"
        params.append(max_lag)

    query += " ORDER BY p_value ASC"
    rows = conn.execute(query, params).fetchall()
    cols = [desc[0] for desc in conn.execute(f"PRAGMA table_info(causal_results)").fetchall()]
    col_names = [c[1] for c in conn.execute("PRAGMA table_info(causal_results)").fetchall()]
    conn.close()

    return {"count": len(rows), "results": [dict(zip(col_names, r)) for r in rows]}


@app.get("/api/v1/models")
def model_comparison():
    """Model comparison from rolling validation."""
    summary_path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        return {
            "stability_reports": data.get("stability_reports", {}),
            "improvements": data.get("improvements", {}),
        }
    return {"error": "No evaluation results found"}


@app.get("/api/v1/drift")
def drift_analysis():
    """Drift and robustness analysis results."""
    summary_path = os.path.join(settings.METRICS_DIR, "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        return {
            "extreme_weather_drift": data.get("extreme_weather_drift", {}),
            "seasonal_drift": data.get("seasonal_drift", []),
        }
    return {"error": "No drift analysis results found"}


@app.get("/api/v1/features/{region}/{need_type}")
def feature_time_series(region: str, need_type: str):
    """Get the feature time series for a specific (region, need_type)."""
    conn = get_raw_connection()
    rows = conn.execute(
        "SELECT * FROM model_features_daily WHERE region_id=? AND need_type=? ORDER BY date",
        (region, need_type),
    ).fetchall()
    col_names = [c[1] for c in conn.execute("PRAGMA table_info(model_features_daily)").fetchall()]
    conn.close()

    return {"count": len(rows), "data": [dict(zip(col_names, r)) for r in rows]}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
