# =============================================================================
# src/api/main.py
# API REST para inferencia del modelo DSMarket
# Requerimiento de Martin (CTO)
# =============================================================================
#
# Cómo arrancar:
#   uvicorn src.api.main:app --reload --port 8000
#
# Documentación interactiva:
#   http://localhost:8000/docs
#
# =============================================================================

from fastapi import FastAPI, HTTPException
from src.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ModelStatusResponse,
    HealthResponse
)
from src.api.service import load_artifacts, predict_from_payload

# =============================================================================
# Inicialización
# =============================================================================

app = FastAPI(
    title="DSMarket Forecast API",
    description=(
        "API de predicción de demanda y reposición de stock para DSMarket. "
        "Modelo base: LightGBM E2 calibrado (MAE 0.97, Bias +0.03%)."
    ),
    version="1.0.0",
)

# Cargar artefactos una sola vez al arrancar el servidor
ARTIFACTS = load_artifacts()


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
def health():
    """Comprueba que el servidor y el modelo están operativos."""
    return {
        "status":       "ok",
        "model_loaded": ARTIFACTS.get("model") is not None
    }


@app.post("/api/v1/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest):
    """
    Predicción de demanda y recomendación de pedido
    para un store × item dado.

    - **store_id**: código de tienda (ej. NYC_3)
    - **item_id**: código de producto
    - **horizon_days**: días a predecir (1-28, default 7)
    - **service_level**: nivel de servicio deseado (0.90-0.99)
    - **current_stock**: stock actual disponible
    - **lag_7 ... has_event**: features del modelo
    """
    try:
        result = predict_from_payload(payload, ARTIFACTS)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/status", response_model=ModelStatusResponse)
def model_status():
    """
    Estado del modelo en producción:
    versión, factor de calibración y métricas actuales.
    """
    return {
        "model_version":      ARTIFACTS["model_version"],
        "calibration_factor": ARTIFACTS["calibration_factor"],
        "mae":                ARTIFACTS["mae"],
        "wape":               ARTIFACTS["wape"],
        "bias":               ARTIFACTS["bias"],
        "status":             "production",
    }