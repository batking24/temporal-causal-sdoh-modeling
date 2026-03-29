-- =============================================================================
-- Temporal Causal Modeling of Climate-Driven Social Needs
-- Full SQLite Schema
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. REGION LOOKUP — shared geographic key across all tables
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS region_lookup (
    region_id   TEXT PRIMARY KEY,          -- canonical key (state-level or county FIPS)
    zipcode     TEXT,
    county_fips TEXT,
    county_name TEXT,
    state       TEXT NOT NULL,
    city        TEXT
);
CREATE INDEX IF NOT EXISTS idx_region_zip   ON region_lookup(zipcode);
CREATE INDEX IF NOT EXISTS idx_region_state ON region_lookup(state);


-- ---------------------------------------------------------------------------
-- 2. RAW TABLES — untouched ingested data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_weather_daily (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,           -- ISO 8601: YYYY-MM-DD
    region_id   TEXT    NOT NULL,
    station_id  TEXT,
    tmax        REAL,                       -- °F
    tmin        REAL,
    tavg        REAL,
    prcp        REAL,                       -- inches
    snow        REAL,                       -- inches
    awnd        REAL,                       -- avg wind speed (mph)
    source      TEXT    DEFAULT 'NOAA_GHCND',
    created_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(date, region_id)
);
CREATE INDEX IF NOT EXISTS idx_raw_wx_date_region ON raw_weather_daily(date, region_id);


CREATE TABLE IF NOT EXISTS raw_social_needs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_id              TEXT,
    ref_code            TEXT,
    ref_status          TEXT,
    ref_date            TEXT    NOT NULL,   -- ISO 8601 datetime
    zipcode             TEXT,
    region_id           TEXT,
    state               TEXT,
    gender              TEXT,
    age                 INTEGER,
    age_group           TEXT,
    language            TEXT,
    ref_type            TEXT,
    lob                 TEXT,              -- line of business (MDCD, MDCR, COMM)
    cohort              TEXT,
    risk_score          REAL,
    category_id         TEXT,
    category            TEXT   NOT NULL,   -- need type (Food Insecurity, etc.)
    need_id             TEXT,
    subcategory         TEXT,
    term_need           TEXT,
    need_source         TEXT,              -- Automatic / Manual
    need_status         TEXT,              -- Confirmed / Unconfirmed / Unmet / …
    need_created_date   TEXT,
    confirmation_date   TEXT,
    days_to_confirm     REAL,
    program_id          TEXT,
    program             TEXT,
    program_status      TEXT,
    program_created_date TEXT,
    source_file         TEXT,              -- tracks which CSV this row came from
    created_at          TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_raw_sn_date     ON raw_social_needs(ref_date);
CREATE INDEX IF NOT EXISTS idx_raw_sn_region   ON raw_social_needs(region_id);
CREATE INDEX IF NOT EXISTS idx_raw_sn_category ON raw_social_needs(category);
CREATE INDEX IF NOT EXISTS idx_raw_sn_state    ON raw_social_needs(state);
CREATE INDEX IF NOT EXISTS idx_raw_sn_zip      ON raw_social_needs(zipcode);


-- ---------------------------------------------------------------------------
-- 3. CLEANED / AGGREGATED TABLES — daily grain
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather_daily_agg (
    date            TEXT    NOT NULL,
    region_id       TEXT    NOT NULL,
    avg_temp        REAL,
    max_temp        REAL,
    min_temp        REAL,
    temp_range      REAL,                  -- max - min
    precip          REAL,
    snow            REAL,
    wind_speed      REAL,
    heatwave_flag   INTEGER DEFAULT 0,     -- 1/0
    coldwave_flag   INTEGER DEFAULT 0,
    heavy_rain_flag INTEGER DEFAULT 0,
    rolling_7d_temp     REAL,
    rolling_14d_temp    REAL,
    rolling_7d_precip   REAL,
    rolling_30d_precip  REAL,
    updated_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, region_id)
);

CREATE TABLE IF NOT EXISTS social_needs_daily_agg (
    date                TEXT    NOT NULL,
    region_id           TEXT    NOT NULL,
    need_type           TEXT    NOT NULL,
    daily_need_count    INTEGER NOT NULL DEFAULT 0,
    confirmed_count     INTEGER DEFAULT 0,
    unmet_count         INTEGER DEFAULT 0,
    rolling_7d_count    REAL,
    rolling_14d_count   REAL,
    rolling_30d_count   REAL,
    updated_at          TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, region_id, need_type)
);


-- ---------------------------------------------------------------------------
-- 4. FEATURE TABLE — the full modeling matrix
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_features_daily (
    date            TEXT    NOT NULL,
    region_id       TEXT    NOT NULL,
    need_type       TEXT    NOT NULL,

    -- target
    target_count    INTEGER NOT NULL,

    -- lagged weather features
    tmax_lag_1      REAL,
    tmax_lag_3      REAL,
    tmax_lag_7      REAL,
    tmax_lag_14     REAL,
    tmax_lag_30     REAL,
    prcp_lag_1      REAL,
    prcp_lag_3      REAL,
    prcp_lag_7      REAL,
    prcp_lag_14     REAL,
    prcp_lag_30     REAL,

    -- rolling weather features
    temp_rollmean_7     REAL,
    temp_rollmean_14    REAL,
    precip_rollsum_7    REAL,
    precip_rollsum_30   REAL,

    -- interaction & trend features
    temp_precip_interact    REAL,
    temp_target_interact    REAL,
    temp_trend_7d           REAL,

    -- lagged target features
    target_lag_1        INTEGER,
    target_lag_7        INTEGER,
    target_rollmean_7   REAL,
    target_rollmean_14  REAL,

    -- event flags
    heatwave_flag       INTEGER,
    coldwave_flag       INTEGER,
    heavy_rain_flag     INTEGER,

    -- calendar features
    day_of_week         INTEGER,   -- 0=Mon, 6=Sun
    month               INTEGER,
    season              TEXT,      -- winter/spring/summer/fall
    holiday_flag        INTEGER,
    is_weekend          INTEGER,

    PRIMARY KEY (date, region_id, need_type)
);


-- ---------------------------------------------------------------------------
-- 5. EXPERIMENT / METRICS / CAUSAL RESULTS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,       -- UUID string
    model_type      TEXT NOT NULL,          -- 'baseline_ar', 'var_weather', 'granger_selected'
    target          TEXT NOT NULL,          -- need type being modeled
    region_scope    TEXT,                   -- 'all', specific state, or region_id
    train_start     TEXT NOT NULL,
    train_end       TEXT NOT NULL,
    test_start      TEXT,
    test_end        TEXT,
    params_json     TEXT,                   -- JSON string of hyperparameters
    description     TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    split_index     INTEGER,               -- rolling validation split number
    rmse            REAL,
    mae             REAL,
    mape            REAL,
    nrmse           REAL,                  -- normalized RMSE
    stability_score REAL,                  -- std dev of RMSE across splits
    forecast_error  REAL,
    variance        REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(experiment_id);

CREATE TABLE IF NOT EXISTS causal_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    feature_name    TEXT NOT NULL,          -- e.g. 'tmax', 'prcp'
    lag             INTEGER NOT NULL,
    p_value         REAL,
    f_statistic     REAL,
    effect_strength REAL,
    significant_flag INTEGER,              -- 1 if p < 0.05
    need_type       TEXT,
    region_id       TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_causal_exp ON causal_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_causal_feature ON causal_results(feature_name, lag);
