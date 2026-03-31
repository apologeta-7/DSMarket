# =============================================================================
# src/api/service.py
# Lógica de negocio de la API — carga de artefactos e inferencia
# =============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from src.paths import (
    MODEL_E2_PATH, METRICS_PATH,
    MODELS_DIR
)
from src.config import (
    CALIBRATION_FACTOR, FEATURE_COLS_E2,
    SERVICE_LEVEL_95, SERVICE_LEVEL_90
)
from src.utils import load_json, calc_safety_stock, calc_order
from src.forecasting_utils import predict_single


# =============================================================================
# Carga de artefactos
# =============================================================================

def load_artifacts() -> dict:
    """
    Carga todos los artefactos necesarios para la inferencia.

    Returns:
        diccionario con modelo, factor de calibración y métricas
    """
    print("⏳ Cargando artefactos del modelo...")

    # Modelo LightGBM
    model = lgb.Booster(model_file=str(MODEL_E2_PATH))
    print(f"✅ Modelo cargado: {MODEL_E2_PATH.name}")

    # Factor de calibración
    cal_path = MODELS_DIR / "calibration_factor.json"
    if cal_path.exists():
        cal_data = load_json(cal_path)
        calibration_factor = cal_data["calibration_factor"]
        model_version      = cal_data.get("version", "E2_calibrado")
    else:
        # Fallback al valor del config si no existe el JSON
        calibration_factor = CALIBRATION_FACTOR
        model_version      = "E2_calibrado"
    print(f"✅ Factor de calibración: ×{calibration_factor:.4f}")

    # Métricas del modelo
    df_metrics = pd.read_csv(METRICS_PATH)
    last_row   = df_metrics[
        df_metrics["model"].str.contains("calibr", case=False)
    ].iloc[-1]

    artifacts = {
        "model":              model,
        "calibration_factor": calibration_factor,
        "model_version":      model_version,
        "mae":                float(last_row["MAE"]),
        "wape":               float(last_row["WAPE (%)"]),
        "bias":               float(last_row["Bias (%)"]),
        "feature_cols":       FEATURE_COLS_E2,
    }

    print("✅ Artefactos listos para inferencia.")
    return artifacts


# =============================================================================
# Inferencia
# =============================================================================

def predict_from_payload(payload, artifacts: dict) -> dict:
    """
    Genera predicción y recomendación de pedido a partir del request.

    Args:
        payload:   ForecastRequest de la API
        artifacts: diccionario devuelto por load_artifacts()

    Returns:
        diccionario con los campos de ForecastResponse
    """
    # Extraer features del payload
    features = {col: getattr(payload, col)
                for col in artifacts["feature_cols"]}

    # Predicción diaria calibrada
    pred_day = predict_single(
        model=artifacts["model"],
        features=features,
        feature_cols=artifacts["feature_cols"],
        calibration_factor=artifacts["calibration_factor"]
    )

    # Escalar al horizonte pedido
    forecast_7d = round(pred_day * payload.horizon_days, 2)

    # Safety stock
    # Usamos rolling_std_7 si está disponible, si no aproximamos
    sigma = features.get("rolling_std_7", features["rolling_mean_7"] * 0.3)
    safety = calc_safety_stock(
        sigma_error=sigma,
        horizon=payload.horizon_days,
        service_level=payload.service_level
    )

    # Pedido recomendado
    order = calc_order(
        forecast_7d=forecast_7d,
        safety_stock=safety,
        current_stock=payload.current_stock
    )

    return {
        "store_id":           payload.store_id,
        "item_id":            payload.item_id,
        "forecast_units_7d":  forecast_7d,
        "safety_stock":       safety,
        "recommended_order":  order,
        "service_level":      payload.service_level,
        "calibration_factor": artifacts["calibration_factor"],
    }