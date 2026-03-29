"""
config.py — Centralised project configuration.

Uses pydantic-settings to load from environment variables / .env file
with sensible defaults for local development.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent          # climate-social-needs-causal/
_BROWN_RESEARCH_ROOT = _PROJECT_ROOT.parent                     # brown_research/


class Settings(BaseSettings):
    """Immutable, validated configuration singleton."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ───────────────────────────────────────────────────────────
    DB_PATH: str = str(_PROJECT_ROOT / "data" / "climate_causal.db")

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.DB_PATH}"

    # ── NOAA API ───────────────────────────────────────────────────────────
    NOAA_API_TOKEN: str = ""               # free token from ncdc.noaa.gov
    NOAA_BASE_URL: str = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

    # ── Paths ──────────────────────────────────────────────────────────────
    PROJECT_ROOT: str = str(_PROJECT_ROOT)
    DATA_DIR: str = str(_PROJECT_ROOT / "data")
    DATA_RAW_DIR: str = str(_PROJECT_ROOT / "data" / "raw")
    DATA_PROCESSED_DIR: str = str(_PROJECT_ROOT / "data" / "processed")
    OUTPUTS_DIR: str = str(_PROJECT_ROOT / "outputs")
    PLOTS_DIR: str = str(_PROJECT_ROOT / "outputs" / "plots")
    METRICS_DIR: str = str(_PROJECT_ROOT / "outputs" / "metrics")
    REPORTS_DIR: str = str(_PROJECT_ROOT / "outputs" / "reports")

    # Path to the GroundGame CSVs the user dropped in brown_research/data/
    SOURCE_DATA_DIR: str = str(_BROWN_RESEARCH_ROOT / "data")

    # ── Modeling ───────────────────────────────────────────────────────────
    MAX_LAG_DAYS: int = 30                 # maximum lag horizon for features
    GRANGER_SIGNIFICANCE: float = 0.05     # p-value threshold
    ROLLING_WINDOW_MONTHS: int = 1         # test window in rolling validation

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (created once per process)."""
    s = Settings()
    # Ensure all output directories exist
    for d in [s.DATA_DIR, s.DATA_RAW_DIR, s.DATA_PROCESSED_DIR,
              s.OUTPUTS_DIR, s.PLOTS_DIR, s.METRICS_DIR, s.REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
    return s
