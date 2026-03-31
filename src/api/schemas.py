# =============================================================================
# src/api/schemas.py
# Modelos Pydantic para request y response de la API
# =============================================================================

from pydantic import BaseModel, Field
from typing import Optional


class ForecastRequest(BaseModel):
    """
    Request para predicción de un único store × item.

    Ejemplo:
        {
            "store_id": "NYC_3",
            "item_id": "PROD_00142",
            "horizon_days": 7,
            "service_level": 0.95,
            "current_stock": 0.0
        }
    """
    store_id:       str   = Field(..., example="NYC_3")
    item_id:        str   = Field(..., example="PROD_00142")
    horizon_days:   int   = Field(default=7, ge=1, le=28)
    service_level:  float = Field(default=0.95, ge=0.80, le=0.99)
    current_stock:  float = Field(default=0.0,  ge=0.0)

    # Features del modelo — valores de los últimos 28 días
    lag_7:               float = Field(..., ge=0)
    lag_14:              float = Field(..., ge=0)
    lag_21:              float = Field(..., ge=0)
    lag_28:              float = Field(..., ge=0)
    rolling_mean_7:      float = Field(..., ge=0)
    rolling_mean_28:     float = Field(..., ge=0)
    day_of_week:         int   = Field(..., ge=0, le=6)
    day_of_month:        int   = Field(..., ge=1, le=31)
    week_of_year:        int   = Field(..., ge=1, le=53)
    month:               int   = Field(..., ge=1, le=12)
    year:                int   = Field(..., ge=2010)
    is_weekend:          int   = Field(..., ge=0, le=1)
    has_event:           int   = Field(..., ge=0, le=1)


class ForecastResponse(BaseModel):
    """
    Response con predicción y recomendación de pedido.

    Ejemplo:
        {
            "store_id": "NYC_3",
            "item_id": "PROD_00142",
            "forecast_units_7d": 140.0,
            "safety_stock": 35.0,
            "recommended_order": 175.0,
            "service_level": 0.95,
            "calibration_factor": 1.3657
        }
    """
    store_id:            str
    item_id:             str
    forecast_units_7d:   float
    safety_stock:        float
    recommended_order:   float
    service_level:       float
    calibration_factor:  float


class ModelStatusResponse(BaseModel):
    """Response con estado del modelo en producción."""
    model_version:        str
    calibration_factor:   float
    mae:                  float
    wape:                 float
    bias:                 float
    status:               str


class HealthResponse(BaseModel):
    """Response del health check."""
    status: str
    model_loaded: bool