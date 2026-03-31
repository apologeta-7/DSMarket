# =============================================================================
# src/config.py
# Configuración global del proyecto DSMarket
# =============================================================================

# Reproducibilidad
SEED = 42

# Columnas clave
TARGET_COL   = "units"
DATE_COL     = "date"
ITEM_COL     = "item"
STORE_COL    = "store_code"
CATEGORY_COL = "category"

ID_COLS = [STORE_COL, ITEM_COL]

# Forecasting
FORECAST_HORIZON = 28   # días
LAG_DAYS         = [7, 14, 21, 28]
ROLLING_WINDOWS  = [7, 28]

# Split temporal
# Test = últimos 28 días del histórico
TEST_DAYS = 28

# Calibración
CALIBRATION_FACTOR = 1.3657

# Clusters de productos
CLUSTER_NAMES = {
    0: "Crecimiento volátil",
    1: "Básicos de alto volumen",
    2: "Premium",
    3: "Estacionales volátiles",
}

# LightGBM — parámetros base E2
LGBM_PARAMS = {
    "objective":         "regression_l1",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

# Features E2 (modelo ganador)
FEATURE_COLS_E2 = [
    "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "day_of_month",
    "week_of_year", "month", "year",
    "is_weekend", "has_event",
]

# Safety stock
SERVICE_LEVEL_95 = 1.65   # Z para 95%
SERVICE_LEVEL_90 = 1.28   # Z para 90%