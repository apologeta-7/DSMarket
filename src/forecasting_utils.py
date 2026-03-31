# =============================================================================
# src/forecasting_utils.py
# Funciones de entrenamiento, evaluación y calibración del modelo
# DSMarket TFM
# =============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from src.config import (
    LGBM_PARAMS, FEATURE_COLS_E2,
    CALIBRATION_FACTOR, SEED,
    STORE_COL, ITEM_COL, DATE_COL, TARGET_COL
)
from src.utils import calc_metrics, save_npy, save_csv


# =============================================================================
# Construcción de arrays X, y para entrenamiento
# =============================================================================

def build_Xy(df_panel: pd.DataFrame,
             mask: pd.Series,
             feature_cols: list) -> tuple:
    """
    Extrae arrays numpy X e y del panel filtrado por máscara.

    Args:
        df_panel:     panel completo store × item × día
        mask:         máscara booleana (is_train o is_test)
        feature_cols: lista de features a usar

    Returns:
        X: array float32 (n_filas × n_features)
        y: array float32 (n_filas,)
    """
    df_filtered = df_panel.loc[mask, feature_cols + [TARGET_COL]].dropna()
    X = df_filtered[feature_cols].values.astype("float32")
    y = df_filtered[TARGET_COL].values.astype("float32")
    print(f"✅ X: {X.shape} | y: {y.shape}")
    return X, y


# =============================================================================
# Entrenamiento de un experimento LightGBM
# =============================================================================

def train_lgbm(X_train: np.ndarray,
               y_train: np.ndarray,
               params: dict = None,
               experiment_name: str = "E") -> lgb.LGBMRegressor:
    """
    Entrena un modelo LightGBM con los parámetros dados.

    Args:
        X_train:         array de features de entrenamiento
        y_train:         array de target de entrenamiento
        params:          parámetros LightGBM (default: LGBM_PARAMS)
        experiment_name: nombre del experimento para el log

    Returns:
        modelo entrenado
    """
    import gc
    if params is None:
        params = LGBM_PARAMS

    print(f"⏳ Entrenando {experiment_name}...")
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    print(f"✅ {experiment_name} entrenado.")

    gc.collect()
    return model


# =============================================================================
# Evaluación de un experimento
# =============================================================================

def evaluate_experiment(model: lgb.LGBMRegressor,
                        X_test: np.ndarray,
                        y_true: np.ndarray,
                        experiment_name: str,
                        models_dir: Path = None,
                        save: bool = True) -> dict:
    """
    Genera predicciones, calcula métricas y opcionalmente guarda artefactos.

    Args:
        model:           modelo LightGBM entrenado
        X_test:          features del período de test
        y_true:          valores reales del test
        experiment_name: nombre del experimento (ej. "E2")
        models_dir:      directorio donde guardar artefactos
        save:            si True, guarda modelo y predicciones en disco

    Returns:
        diccionario de métricas
    """
    preds = np.clip(model.predict(X_test), 0, None).astype("float32")
    metrics = calc_metrics(y_true, preds, experiment_name)

    print(f"\n📊 {experiment_name} — Métricas:")
    for k, v in metrics.items():
        if k != "model":
            print(f"   {k}: {v}")

    if save and models_dir is not None:
        models_dir = Path(models_dir)
        # Guardar predicciones
        save_npy(preds, models_dir / f"preds_{experiment_name}.npy")
        # Guardar modelo serializado
        model_path = str(models_dir / f"model_{experiment_name}.txt")
        model.booster_.save_model(model_path)
        print(f"✅ Modelo guardado: {model_path}")

    return metrics


# =============================================================================
# Calibración global
# =============================================================================

def calibrate_predictions(preds: np.ndarray,
                           y_true: np.ndarray,
                           experiment_name: str = "E2_calibrado") -> tuple:
    """
    Aplica calibración global para corregir el bias sistemático.

    Calcula el ratio predicho/real y aplica el factor inverso.
    En producción este factor se recalcula mensualmente.

    Args:
        preds:           predicciones sin calibrar
        y_true:          valores reales
        experiment_name: nombre para el log

    Returns:
        preds_cal:          predicciones calibradas
        calibration_factor: factor aplicado (float)
        metrics_cal:        métricas post-calibración
    """
    # Calcular ratio predicho/real
    ratio = (preds.sum() / (y_true.sum() + 1e-9))
    calibration_factor = 1 / ratio

    print(f"📊 Ratio predicho/real: {ratio:.4f}")
    print(f"📊 Factor de calibración: ×{calibration_factor:.4f}")

    # Aplicar calibración
    preds_cal = np.clip(preds * calibration_factor, 0, None)
    metrics_cal = calc_metrics(y_true, preds_cal, experiment_name)

    print(f"\n📊 Métricas post-calibración:")
    for k, v in metrics_cal.items():
        if k != "model":
            print(f"   {k}: {v}")

    return preds_cal, calibration_factor, metrics_cal


# =============================================================================
# Calibración por categoría
# =============================================================================

def calibrate_by_category(df_test_diag: pd.DataFrame,
                           preds: np.ndarray) -> pd.DataFrame:
    """
    Calcula el factor de calibración por categoría (E7 del roadmap).

    Corrige el bias diferencial por categoría:
        ACCESSORIES:  ~-45%
        HOME&GARDEN:  ~-34%
        SUPERMARKET:  ~-21%

    Args:
        df_test_diag: DataFrame del período de test con columna 'category'
        preds:        predicciones sin calibrar

    Returns:
        DataFrame con columnas [category, ratio, calibration_factor]
    """
    df = df_test_diag.copy()
    df["pred"] = preds

    df_cat = (
        df.groupby("category")[["units", "pred"]]
        .sum()
        .reset_index()
    )
    df_cat["ratio"]              = (df_cat["pred"] / (df_cat["units"] + 1e-9)).round(4)
    df_cat["calibration_factor"] = (1 / df_cat["ratio"]).round(4)

    print("📊 Factores de calibración por categoría:")
    print(df_cat[["category", "ratio", "calibration_factor"]].to_string(index=False))

    return df_cat


# =============================================================================
# Tabla comparativa de experimentos
# =============================================================================

def build_metrics_table(metrics_list: list,
                        save_path: Path = None) -> pd.DataFrame:
    """
    Construye tabla comparativa de métricas de todos los experimentos.

    Args:
        metrics_list: lista de dicts devueltos por calc_metrics
        save_path:    si se indica, guarda el CSV

    Returns:
        DataFrame ordenado por MAE ascendente
    """
    df = pd.DataFrame(metrics_list)
    df = df.sort_values("MAE").reset_index(drop=True)

    if save_path is not None:
        save_csv(df, save_path)

    return df


# =============================================================================
# Inferencia sobre nuevos datos (producción)
# =============================================================================

def predict_single(model: lgb.LGBMRegressor,
                   features: dict,
                   feature_cols: list,
                   calibration_factor: float = CALIBRATION_FACTOR) -> float:
    """
    Genera una predicción para un único store × item.
    Usado por la API FastAPI.

    Args:
        model:              modelo LightGBM cargado
        features:           diccionario con valores de las features
        feature_cols:       lista ordenada de features del modelo
        calibration_factor: factor de calibración a aplicar

    Returns:
        predicción calibrada en unidades (float)
    """
    x = np.array([[features[col] for col in feature_cols]],
                 dtype="float32")
    pred = float(np.clip(model.predict(x), 0, None)[0])
    pred_cal = round(pred * calibration_factor, 2)
    return pred_cal