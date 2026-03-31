# =============================================================================
# src/preprocessing.py
# Funciones de limpieza, transformación y construcción del panel
# DSMarket TFM
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import (
    ITEM_COL, STORE_COL, CATEGORY_COL,
    DATE_COL, TARGET_COL,
    LAG_DAYS, ROLLING_WINDOWS, SEED
)


# =============================================================================
# Construcción del panel store × item × día
# =============================================================================

def build_panel(df_sales: pd.DataFrame,
                df_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte df_sales de formato ancho a formato largo (panel).
    """
    day_cols  = [c for c in df_sales.columns if c.startswith("d_")]
    meta_cols = [ITEM_COL, STORE_COL, CATEGORY_COL]

    print(f"⏳ Construyendo panel: {len(day_cols):,} días × "
          f"{df_sales.shape[0]:,} series...")

    df_panel = df_sales[meta_cols + day_cols].melt(
        id_vars=meta_cols,
        value_vars=day_cols,
        var_name="d",
        value_name=TARGET_COL
    )

    df_cal_slim = df_cal[["d", DATE_COL, "weekday_int"]].copy()
    df_panel = df_panel.merge(df_cal_slim, on="d", how="left")
    df_panel = df_panel.sort_values(
        [STORE_COL, ITEM_COL, DATE_COL]
    ).reset_index(drop=True)

    n_expected = df_sales.shape[0] * len(day_cols)
    assert len(df_panel) == n_expected, \
        f"❌ Filas esperadas {n_expected:,} vs obtenidas {len(df_panel):,}"
    assert df_panel[TARGET_COL].isna().sum() == 0, \
        "❌ Nulos en columna units"

    print(f"✅ Panel construido: {len(df_panel):,} filas")
    print(f"   Rango fechas: {df_panel[DATE_COL].min().date()} → "
          f"{df_panel[DATE_COL].max().date()}")
    print(f"   Memoria: {df_panel.memory_usage(deep=True).sum()/1e6:.1f} MB")

    return df_panel


# =============================================================================
# Split temporal
# =============================================================================

def temporal_split(df_panel: pd.DataFrame,
                   test_days: int = 28):
    """
    Define máscaras de train y test sin leakage.
    """
    max_date   = df_panel[DATE_COL].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    train_end  = test_start - pd.Timedelta(days=1)

    is_train = df_panel[DATE_COL] <= train_end
    is_test  = df_panel[DATE_COL] >= test_start

    print(f"📅 Train: hasta {train_end.date()} ({is_train.sum():,} filas)")
    print(f"📅 Test:  {test_start.date()} → {max_date.date()} ({is_test.sum():,} filas)")

    return is_train, is_test, train_end, test_start


# =============================================================================
# Feature engineering — lags
# =============================================================================

def add_lag_features(df_panel: pd.DataFrame,
                     lags: list = LAG_DAYS) -> pd.DataFrame:
    """
    Añade lags de ventas por store × item.
    """
    print(f"⏳ Calculando lags {lags}...")

    df_panel = df_panel.sort_values([STORE_COL, ITEM_COL, DATE_COL])

    for lag in lags:
        df_panel[f"lag_{lag}"] = (
            df_panel
            .groupby([STORE_COL, ITEM_COL], observed=True)[TARGET_COL]
            .shift(lag)
        )

    print(f"✅ Lags añadidos: {[f'lag_{l}' for l in lags]}")
    return df_panel


# =============================================================================
# Feature engineering — rolling features
# =============================================================================

def add_rolling_features(df_panel: pd.DataFrame,
                         windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Añade rolling mean, std, max y min por store × item.
    """
    print(f"⏳ Calculando rolling features {windows}...")

    for window in windows:
        grp = df_panel.groupby(
            [STORE_COL, ITEM_COL], observed=True
        )[TARGET_COL]

        df_panel[f"rolling_mean_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df_panel[f"rolling_std_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    grp28 = df_panel.groupby([STORE_COL, ITEM_COL], observed=True)[TARGET_COL]

    df_panel["rolling_max_28"] = grp28.transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).max()
    )
    df_panel["rolling_min_28"] = grp28.transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).min()
    )

    print("✅ Rolling features añadidas")
    return df_panel


# =============================================================================
# Feature engineering — calendario
# =============================================================================

def add_calendar_features(df_panel: pd.DataFrame,
                          df_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features temporales y de eventos al panel.
    """
    print("⏳ Añadiendo features de calendario...")

    df_panel["day_of_week"]  = df_panel[DATE_COL].dt.dayofweek
    df_panel["day_of_month"] = df_panel[DATE_COL].dt.day
    df_panel["week_of_year"] = (
        df_panel[DATE_COL].dt.isocalendar().week.astype(int)
    )
    df_panel["month"] = df_panel[DATE_COL].dt.month
    df_panel["year"]  = df_panel[DATE_COL].dt.year
    df_panel["is_weekend"] = (df_panel["day_of_week"] >= 5).astype(int)

    df_cal_event = df_cal[["date", "event"]].copy()
    df_cal_event["has_event"] = df_cal_event["event"].notna().astype(int)
    df_panel = df_panel.merge(
        df_cal_event[["date", "has_event"]],
        on="date",
        how="left"
    )
    df_panel["has_event"] = df_panel["has_event"].fillna(0).astype(int)

    print("✅ Features de calendario añadidas")
    return df_panel


# =============================================================================
# Feature engineering — precios (E6)
# =============================================================================

def add_price_features(df_panel: pd.DataFrame,
                       df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features de precio al panel (E6 del roadmap).

    Features añadidas:
        sell_price, price_lag_1, price_change,
        price_change_pct, price_rolling_mean_4w, price_ratio_vs_avg
    """
    print("⏳ Añadiendo features de precio...")

    df_p = df_prices.copy()

    # Limpiar yearweek: eliminar NaN e infinitos
    yearweek_num = pd.to_numeric(df_p["yearweek"], errors="coerce")
    invalid_mask = yearweek_num.isna()
    if invalid_mask.any():
        print(f"   ⚠️ yearweek inválido: {invalid_mask.sum():,} filas omitidas")
    df_p = df_p.loc[~invalid_mask].copy()
    df_p["yearweek"] = (
        yearweek_num.loc[~invalid_mask]
        .astype(int)
        .astype(str)
        .str.zfill(6)
    )

    # Convertir yearweek a fecha
    df_p["date_week"] = pd.to_datetime(
        df_p["yearweek"] + "-1",
        format="%Y%W-%w",
        errors="coerce",
    )

    # Eliminar fechas que no se pudieron parsear
    bad_date = df_p["date_week"].isna()
    if bad_date.any():
        print(f"   ⚠️ formato inválido: {bad_date.sum():,} filas omitidas")
    df_p = df_p.loc[~bad_date].copy()

    # Precio medio por item × semana
    df_price_weekly = (
        df_p.groupby([ITEM_COL, "date_week"])["sell_price"]
        .mean()
        .reset_index()
        .rename(columns={"date_week": DATE_COL})
    )

    # Lags de precio
    df_price_weekly = df_price_weekly.sort_values([ITEM_COL, DATE_COL])
    df_price_weekly["price_lag_1"] = (
        df_price_weekly.groupby(ITEM_COL)["sell_price"].shift(1)
    )
    df_price_weekly["price_change"] = (
        df_price_weekly["sell_price"] - df_price_weekly["price_lag_1"]
    )
    df_price_weekly["price_change_pct"] = (
        df_price_weekly["price_change"]
        / (df_price_weekly["price_lag_1"] + 1e-9) * 100
    ).round(4)

    # Rolling mean 4 semanas
    df_price_weekly["price_rolling_mean_4w"] = (
        df_price_weekly
        .groupby(ITEM_COL)["sell_price"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )

    # Ratio precio actual vs media histórica
    price_avg = (
        df_price_weekly.groupby(ITEM_COL)["sell_price"]
        .mean()
        .rename("price_avg_global")
        .reset_index()
    )
    df_price_weekly = df_price_weekly.merge(price_avg, on=ITEM_COL, how="left")
    df_price_weekly["price_ratio_vs_avg"] = (
        df_price_weekly["sell_price"]
        / (df_price_weekly["price_avg_global"] + 1e-9)
    ).round(4)

    # Añadir semana al panel para el join
    df_panel["date_week"] = df_panel[DATE_COL] - pd.to_timedelta(
        df_panel[DATE_COL].dt.dayofweek, unit="d"
    )

    price_cols = [
        ITEM_COL, DATE_COL, "sell_price",
        "price_lag_1", "price_change", "price_change_pct",
        "price_rolling_mean_4w", "price_ratio_vs_avg"
    ]
    df_panel = df_panel.merge(
        df_price_weekly[price_cols],
        left_on=[ITEM_COL, "date_week"],
        right_on=[ITEM_COL, DATE_COL],
        how="left"
    )

    # Limpiar columnas duplicadas del merge
    if DATE_COL + "_y" in df_panel.columns:
        df_panel = df_panel.drop(columns=[DATE_COL + "_y"])
        df_panel = df_panel.rename(columns={DATE_COL + "_x": DATE_COL})

    # Imputar nulos con mediana por item
    for col in ["sell_price", "price_lag_1", "price_change",
                "price_change_pct", "price_rolling_mean_4w",
                "price_ratio_vs_avg"]:
        df_panel[col] = df_panel[col].fillna(
            df_panel.groupby(ITEM_COL)[col].transform("median")
        )

    print("✅ Features de precio añadidas: sell_price, price_lag_1, "
          "price_change, price_change_pct, price_rolling_mean_4w, "
          "price_ratio_vs_avg")
    return df_panel


# =============================================================================
# Feature engineering — clusters
# =============================================================================

def add_cluster_features(df_panel: pd.DataFrame,
                         df_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Une los clusters de producto al panel.
    """
    df_cluster_slim = df_clusters[[ITEM_COL, "cluster", "cluster_name"]].copy()
    df_panel = df_panel.merge(df_cluster_slim, on=ITEM_COL, how="left")
    df_panel["cluster"] = df_panel["cluster"].fillna(-1).astype(int)

    print(f"✅ Clusters añadidos: {df_panel['cluster'].nunique()} grupos")
    return df_panel
