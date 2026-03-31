# =============================================================================
# src/utils.py
# Funciones de utilidad general — DSMarket TFM
# =============================================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import SEED


# =============================================================================
# Reproducibilidad
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    """Fija el seed global para reproducibilidad."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    print(f"✅ Seed fijado: {seed}")


# =============================================================================
# Diagnóstico de DataFrames
# =============================================================================

def df_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Resumen rápido de un DataFrame: shape, memoria, nulos, dtypes."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape:    {df.shape[0]:,} filas × {df.shape[1]:,} cols")
    print(f"  Memoria:  {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    total_nulls = df.isnull().sum().sum()
    print(f"  Nulos:    {total_nulls:,} ({total_nulls/df.size*100:.2f}%)")
    print(f"  Dtypes:   {dict(df.dtypes.value_counts())}")
    print(f"{'='*60}\n")


def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve tabla de columnas con nulos y su porcentaje."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) == 0:
        print("✅ Sin nulos.")
        return pd.DataFrame()
    result = pd.DataFrame({
        "n_nulls": null_counts,
        "pct":     (null_counts / len(df) * 100).round(2)
    }).sort_values("pct", ascending=False)
    return result


# =============================================================================
# Métricas de forecasting
# =============================================================================

def calc_metrics(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 model_name: str) -> dict:
    """
    Calcula métricas técnicas y de negocio para forecasting.
    MAE, RMSE, WAPE, Bias, Underforecast Rate.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true + 1e-9) * 100
    bias = np.sum(y_pred - y_true) / np.sum(y_true + 1e-9) * 100
    underforecast_rate = np.mean(y_pred < y_true) * 100

    return {
        "model":             model_name,
        "MAE":               round(mae, 4),
        "RMSE":              round(rmse, 4),
        "WAPE (%)":          round(wape, 2),
        "Bias (%)":          round(bias, 2),
        "Underforecast (%)": round(underforecast_rate, 2),
    }


# =============================================================================
# Guardado seguro de artefactos
# =============================================================================

def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Guarda DataFrame como CSV creando el directorio si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    print(f"✅ CSV guardado: {path}")


def save_json(obj: dict, path: Path) -> None:
    """Guarda diccionario como JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"✅ JSON guardado: {path}")


def load_json(path: Path) -> dict:
    """Carga JSON desde disco."""
    with open(Path(path)) as f:
        return json.load(f)


def save_npy(arr: np.ndarray, path: Path) -> None:
    """Guarda array numpy en disco."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)
    print(f"✅ NPY guardado: {path}")


# =============================================================================
# Reducción de memoria
# =============================================================================

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce uso de RAM convirtiendo float64→float32 e int64→int16.
    Útil antes de entrenar modelos sobre datasets grandes.
    """
    mem_before = df.memory_usage(deep=True).sum() / 1e6
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int16")
    mem_after = df.memory_usage(deep=True).sum() / 1e6
    print(f"📊 Memoria: {mem_before:.1f} MB → {mem_after:.1f} MB "
          f"({(1 - mem_after/mem_before)*100:.1f}% reducción)")
    return df


# =============================================================================
# Safety stock
# =============================================================================

def calc_safety_stock(sigma_error: float,
                      horizon: int = 7,
                      service_level: float = 0.95) -> float:
    """
    Calcula el safety stock para un horizonte dado.

    Fórmula: Z × σ_error × √horizon

    Args:
        sigma_error:   desviación estándar del error histórico del modelo
        horizon:       número de días del horizonte (default 7)
        service_level: nivel de servicio deseado (0.90 o 0.95)

    Returns:
        safety_stock en unidades
    """
    from src.config import SERVICE_LEVEL_95, SERVICE_LEVEL_90
    z = SERVICE_LEVEL_95 if service_level >= 0.95 else SERVICE_LEVEL_90
    return round(z * sigma_error * np.sqrt(horizon), 2)


def calc_order(forecast_7d: float,
               safety_stock: float,
               current_stock: float = 0.0) -> float:
    """
    Calcula el pedido semanal.

    Fórmula: max(0, predicción_7días + safety_stock - stock_actual)

    Args:
        forecast_7d:    predicción de demanda para los próximos 7 días
        safety_stock:   colchón de seguridad en unidades
        current_stock:  stock disponible actualmente (default 0)

    Returns:
        pedido recomendado en unidades (nunca negativo)
    """
    return max(0.0, round(forecast_7d + safety_stock - current_stock, 2))