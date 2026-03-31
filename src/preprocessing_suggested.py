# =============================================================================
# src/preprocessing.py
# Robust preprocessing utilities for DSMarket forecasting
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ITEM_COL,
    STORE_COL,
    CATEGORY_COL,
    DATE_COL,
    TARGET_COL,
    LAG_DAYS,
    ROLLING_WINDOWS,
    TEST_DAYS,
)

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    """Validate that all required columns exist."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _get_day_columns(df_sales: pd.DataFrame) -> list[str]:
    """Return sorted day columns like d_1, d_2, ..., d_n."""
    day_cols = [c for c in df_sales.columns if str(c).startswith("d_")]
    if not day_cols:
        raise ValueError("No day columns found in sales data (expected columns like d_1, d_2, ...).")

    day_cols = sorted(day_cols, key=lambda x: int(str(x).split("_")[1]))
    return day_cols


def _reduce_panel_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and keep categories compact when possible."""
    df = df.copy()

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def _build_iso_yearweek_from_date(date_series: pd.Series) -> pd.Series:
    """
    Build ISO-compatible yearweek integer from datetime series.
    Example: 2013-W28 -> 201328
    """
    iso = date_series.dt.isocalendar()
    return (iso["year"].astype("int32") * 100 + iso["week"].astype("int32")).astype("int32")


# -----------------------------------------------------------------------------
# Raw input validation
# -----------------------------------------------------------------------------

def validate_sales_input(df_sales: pd.DataFrame) -> None:
    """Validate raw sales input."""
    required = [ITEM_COL, STORE_COL, CATEGORY_COL]
    _check_required_columns(df_sales, required, "df_sales")

    day_cols = _get_day_columns(df_sales)
    if len(day_cols) == 0:
        raise ValueError("df_sales does not contain daily columns.")

    dup_count = df_sales.duplicated(subset=[STORE_COL, ITEM_COL]).sum()
    if dup_count > 0:
        raise ValueError(f"df_sales has {dup_count:,} duplicated store × item rows.")


def validate_calendar_input(df_cal: pd.DataFrame) -> None:
    """Validate raw calendar input."""
    required = ["d", DATE_COL]
    _check_required_columns(df_cal, required, "df_cal")

    dup_count = df_cal.duplicated(subset=["d"]).sum()
    if dup_count > 0:
        raise ValueError(f"df_cal has {dup_count:,} duplicated 'd' keys.")

    if not pd.api.types.is_datetime64_any_dtype(df_cal[DATE_COL]):
        raise ValueError("df_cal['date'] must be datetime64.")

    if df_cal[DATE_COL].isna().any():
        raise ValueError("df_cal['date'] contains null values.")


def validate_prices_input(df_prices: pd.DataFrame) -> None:
    """Validate raw prices input."""
    required = [ITEM_COL, STORE_COL, "yearweek", "sell_price"]
    _check_required_columns(df_prices, required, "df_prices")


def validate_clusters_input(df_clusters: pd.DataFrame) -> None:
    """Validate optional clusters input."""
    required = [ITEM_COL, "cluster"]
    _check_required_columns(df_clusters, required, "df_clusters")


# -----------------------------------------------------------------------------
# Calendar preparation
# -----------------------------------------------------------------------------

def prepare_calendar(df_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize calendar table and create a clean event flag.
    Expected columns:
        - d
        - date
        - optional: event
    """
    validate_calendar_input(df_cal)

    df = df_cal.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError("Calendar contains invalid dates after parsing.")

    if "event" not in df.columns:
        df["event"] = np.nan

    df["has_event"] = df["event"].notna().astype("int8")
    df["day_of_week"] = df[DATE_COL].dt.dayofweek.astype("int8")
    df["day_of_month"] = df[DATE_COL].dt.day.astype("int8")
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype("int16")
    df["month"] = df[DATE_COL].dt.month.astype("int8")
    df["year"] = df[DATE_COL].dt.year.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["yearweek"] = _build_iso_yearweek_from_date(df[DATE_COL])

    keep_cols = [
        "d",
        DATE_COL,
        "event",
        "has_event",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "year",
        "is_weekend",
        "yearweek",
    ]

    return df[keep_cols].copy()


# -----------------------------------------------------------------------------
# Prices preparation
# -----------------------------------------------------------------------------

def prepare_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price table for a safe merge at store × item × yearweek.
    """
    validate_prices_input(df_prices)

    df = df_prices.copy()

    # Normalize yearweek to integer
    df["yearweek"] = pd.to_numeric(df["yearweek"], errors="coerce")
    bad_yearweek = df["yearweek"].isna().sum()
    if bad_yearweek > 0:
        print(f"⚠️ Dropping {bad_yearweek:,} price rows with invalid yearweek.")
        df = df.loc[df["yearweek"].notna()].copy()

    df["yearweek"] = df["yearweek"].astype("int32")

    # Normalize price type
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce")
    bad_price = df["sell_price"].isna().sum()
    if bad_price > 0:
        print(f"⚠️ Dropping {bad_price:,} price rows with invalid sell_price.")
        df = df.loc[df["sell_price"].notna()].copy()

    df["sell_price"] = df["sell_price"].astype("float32")

    # Keep category if available, but do not use it as a join key
    keep_cols = [ITEM_COL, STORE_COL, "yearweek", "sell_price"]
    if CATEGORY_COL in df.columns:
        keep_cols.append(CATEGORY_COL)

    df = df[keep_cols].copy()

    # One row per store × item × yearweek
    df = (
        df.groupby([STORE_COL, ITEM_COL, "yearweek"], observed=True, as_index=False)["sell_price"]
        .mean()
    )
    df["sell_price"] = df["sell_price"].astype("float32")

    dup_count = df.duplicated(subset=[STORE_COL, ITEM_COL, "yearweek"]).sum()
    if dup_count > 0:
        raise ValueError(f"Prepared prices still contain {dup_count:,} duplicated keys.")

    return df


# -----------------------------------------------------------------------------
# Panel construction
# -----------------------------------------------------------------------------

def build_panel(df_sales: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily panel at store × item × date from wide sales table.
    """
    validate_sales_input(df_sales)

    cal = prepare_calendar(df_cal)
    day_cols = _get_day_columns(df_sales)

    # Keep metadata that is known to exist in the raw sales table
    meta_cols = [ITEM_COL, STORE_COL]
    optional_meta = [CATEGORY_COL, "city"]
    for col in optional_meta:
        if col in df_sales.columns:
            meta_cols.append(col)

    print(
        f"⏳ Building panel from sales: {df_sales.shape[0]:,} series × "
        f"{len(day_cols):,} days..."
    )

    df_panel = df_sales[meta_cols + day_cols].melt(
        id_vars=meta_cols,
        value_vars=day_cols,
        var_name="d",
        value_name=TARGET_COL,
    )

    df_panel = df_panel.merge(
        cal,
        on="d",
        how="left",
        validate="many_to_one",
    )

    if df_panel[DATE_COL].isna().any():
        missing_dates = df_panel[DATE_COL].isna().sum()
        raise ValueError(f"Panel merge with calendar left {missing_dates:,} rows without date.")

    expected_rows = df_sales.shape[0] * len(day_cols)
    if len(df_panel) != expected_rows:
        raise ValueError(
            f"Panel row count mismatch. Expected {expected_rows:,}, got {len(df_panel):,}."
        )

    # Ensure target is numeric
    df_panel[TARGET_COL] = pd.to_numeric(df_panel[TARGET_COL], errors="coerce")
    if df_panel[TARGET_COL].isna().any():
        bad_target = df_panel[TARGET_COL].isna().sum()
        raise ValueError(f"Target column '{TARGET_COL}' contains {bad_target:,} invalid values.")

    df_panel[TARGET_COL] = df_panel[TARGET_COL].astype("float32")

    # Add ISO yearweek at daily grain for weekly joins
    df_panel["yearweek"] = _build_iso_yearweek_from_date(df_panel[DATE_COL])

    # Final sort
    df_panel = df_panel.sort_values([STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)

    # Check uniqueness at final grain
    dup_count = df_panel.duplicated(subset=[STORE_COL, ITEM_COL, DATE_COL]).sum()
    if dup_count > 0:
        raise ValueError(f"Panel has {dup_count:,} duplicated store × item × date rows.")

    print(f"✅ Panel built: {len(df_panel):,} rows")
    print(f"   Date range: {df_panel[DATE_COL].min().date()} -> {df_panel[DATE_COL].max().date()}")

    return _reduce_panel_dtypes(df_panel)


# -----------------------------------------------------------------------------
# Calendar features
# -----------------------------------------------------------------------------

def add_calendar_features(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features are already prepared inside build_panel().
    This function only validates their presence and returns the panel.
    """
    required = [
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "year",
        "is_weekend",
        "has_event",
    ]
    _check_required_columns(df_panel, required, "df_panel")
    return df_panel.copy()


# -----------------------------------------------------------------------------
# Lag features
# -----------------------------------------------------------------------------

def add_lag_features(df_panel: pd.DataFrame, lags: list[int] = LAG_DAYS) -> pd.DataFrame:
    """
    Add lag features by store × item series.
    """
    df = df_panel.copy()
    df = df.sort_values([STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)

    print(f"⏳ Adding lag features: {lags}")

    grp = df.groupby([STORE_COL, ITEM_COL], observed=True)[TARGET_COL]
    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag).astype("float32")

    return _reduce_panel_dtypes(df)


# -----------------------------------------------------------------------------
# Rolling features
# -----------------------------------------------------------------------------

def add_rolling_features(
    df_panel: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS,
    add_std: bool = False,
    add_minmax_28: bool = False,
) -> pd.DataFrame:
    """
    Add rolling features without leakage by shifting first.
    """
    df = df_panel.copy()
    df = df.sort_values([STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)

    print(f"⏳ Adding rolling features: {windows}")

    grp = df.groupby([STORE_COL, ITEM_COL], observed=True)[TARGET_COL]

    for window in windows:
        shifted = grp.shift(1)

        df[f"rolling_mean_{window}"] = (
            shifted.groupby([df[STORE_COL], df[ITEM_COL]], observed=True)
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
            .astype("float32")
        )

        if add_std:
            df[f"rolling_std_{window}"] = (
                shifted.groupby([df[STORE_COL], df[ITEM_COL]], observed=True)
                .transform(lambda x: x.rolling(window, min_periods=2).std())
                .astype("float32")
            )

    if add_minmax_28:
        shifted = grp.shift(1)
        df["rolling_max_28"] = (
            shifted.groupby([df[STORE_COL], df[ITEM_COL]], observed=True)
            .transform(lambda x: x.rolling(28, min_periods=1).max())
            .astype("float32")
        )
        df["rolling_min_28"] = (
            shifted.groupby([df[STORE_COL], df[ITEM_COL]], observed=True)
            .transform(lambda x: x.rolling(28, min_periods=1).min())
            .astype("float32")
        )

    return _reduce_panel_dtypes(df)


# -----------------------------------------------------------------------------
# Price features
# -----------------------------------------------------------------------------

def add_price_features(
    df_panel: pd.DataFrame,
    df_prices: pd.DataFrame,
    impute_strategy: str = "store_item_median",
) -> pd.DataFrame:
    """
    Merge weekly prices into the daily panel using store × item × yearweek.

    Added columns:
        - sell_price
        - price_lag_1w
        - price_change
        - price_change_pct
        - price_rolling_mean_4w
        - price_ratio_vs_avg
    """
    df = df_panel.copy()
    df_p = prepare_prices(df_prices)

    print("⏳ Merging price features at store × item × yearweek...")

    # Price history at weekly grain
    df_p = df_p.sort_values([STORE_COL, ITEM_COL, "yearweek"]).reset_index(drop=True)

    grp = df_p.groupby([STORE_COL, ITEM_COL], observed=True)["sell_price"]

    df_p["price_lag_1w"] = grp.shift(1).astype("float32")
    df_p["price_change"] = (df_p["sell_price"] - df_p["price_lag_1w"]).astype("float32")
    df_p["price_change_pct"] = (
        100 * df_p["price_change"] / (df_p["price_lag_1w"] + 1e-9)
    ).astype("float32")

    df_p["price_rolling_mean_4w"] = (
        grp.shift(1)
        .groupby([df_p[STORE_COL], df_p[ITEM_COL]], observed=True)
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
        .astype("float32")
    )

    df_p["price_avg_global"] = grp.transform("mean").astype("float32")
    df_p["price_ratio_vs_avg"] = (
        df_p["sell_price"] / (df_p["price_avg_global"] + 1e-9)
    ).astype("float32")

    price_feature_cols = [
        STORE_COL,
        ITEM_COL,
        "yearweek",
        "sell_price",
        "price_lag_1w",
        "price_change",
        "price_change_pct",
        "price_rolling_mean_4w",
        "price_ratio_vs_avg",
    ]

    df = df.merge(
        df_p[price_feature_cols],
        on=[STORE_COL, ITEM_COL, "yearweek"],
        how="left",
        validate="many_to_one",
    )

    missing_sell_price = df["sell_price"].isna().mean() * 100
    print(f"   Missing sell_price after merge: {missing_sell_price:.2f}%")

    # Safe imputations
    price_cols = [
        "sell_price",
        "price_lag_1w",
        "price_change",
        "price_change_pct",
        "price_rolling_mean_4w",
        "price_ratio_vs_avg",
    ]

    if impute_strategy == "store_item_median":
        for col in price_cols:
            df[col] = df[col].fillna(
                df.groupby([STORE_COL, ITEM_COL], observed=True)[col].transform("median")
            )

    elif impute_strategy == "item_median":
        for col in price_cols:
            df[col] = df[col].fillna(
                df.groupby(ITEM_COL, observed=True)[col].transform("median")
            )

    elif impute_strategy == "global":
        for col in price_cols:
            df[col] = df[col].fillna(df[col].median())

    else:
        raise ValueError(
            "impute_strategy must be one of: "
            "['store_item_median', 'item_median', 'global']"
        )

    # Final fallback for any remaining edge cases
    for col in price_cols:
        df[col] = df[col].fillna(0).astype("float32")

    return _reduce_panel_dtypes(df)


# -----------------------------------------------------------------------------
# Cluster features
# -----------------------------------------------------------------------------

def add_cluster_features(df_panel: pd.DataFrame, df_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Merge item-level clusters into the daily panel.
    Expected columns:
        - item
        - cluster
        - optional: cluster_name
    """
    validate_clusters_input(df_clusters)

    df = df_panel.copy()
    keep_cols = [ITEM_COL, "cluster"]
    if "cluster_name" in df_clusters.columns:
        keep_cols.append("cluster_name")

    cluster_df = df_clusters[keep_cols].drop_duplicates(subset=[ITEM_COL]).copy()

    df = df.merge(
        cluster_df,
        on=ITEM_COL,
        how="left",
        validate="many_to_one",
    )

    df["cluster"] = df["cluster"].fillna(-1).astype("int16")
    if "cluster_name" in df.columns:
        df["cluster_name"] = df["cluster_name"].fillna("Unknown")

    return df


# -----------------------------------------------------------------------------
# Final dataset checks
# -----------------------------------------------------------------------------

def validate_model_panel(df_panel: pd.DataFrame, feature_cols: list[str] | None = None) -> None:
    """
    Validate final panel before training.
    """
    required = [STORE_COL, ITEM_COL, DATE_COL, TARGET_COL]
    _check_required_columns(df_panel, required, "df_panel")

    dup_count = df_panel.duplicated(subset=[STORE_COL, ITEM_COL, DATE_COL]).sum()
    if dup_count > 0:
        raise ValueError(f"Final panel has {dup_count:,} duplicated store × item × date rows.")

    if feature_cols is not None:
        _check_required_columns(df_panel, feature_cols, "df_panel")

    print("✅ Final panel validation passed.")


# -----------------------------------------------------------------------------
# Temporal split
# -----------------------------------------------------------------------------

def temporal_split(df_panel: pd.DataFrame, test_days: int = TEST_DAYS) -> tuple[pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    """
    Create leakage-safe temporal split.
    Test = last `test_days` days of the panel.
    """
    max_date = df_panel[DATE_COL].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    train_end = test_start - pd.Timedelta(days=1)

    is_train = df_panel[DATE_COL] <= train_end
    is_test = df_panel[DATE_COL] >= test_start

    print(f"📅 Train end:  {train_end.date()} | rows: {is_train.sum():,}")
    print(f"📅 Test start: {test_start.date()} | rows: {is_test.sum():,}")

    return is_train, is_test, train_end, test_start


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def build_forecasting_panel(
    df_sales: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_prices: pd.DataFrame | None = None,
    df_clusters: pd.DataFrame | None = None,
    include_price_features: bool = False,
    include_cluster_features: bool = False,
    include_rolling_std: bool = False,
    include_minmax_28: bool = False,
) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline for forecasting.

    Default behavior is aligned with the current E2 setup:
        - panel
        - calendar features
        - lags
        - rolling means
        - no price features
        - no clusters
    """
    df_panel = build_panel(df_sales=df_sales, df_cal=df_cal)
    df_panel = add_calendar_features(df_panel)
    df_panel = add_lag_features(df_panel, lags=LAG_DAYS)
    df_panel = add_rolling_features(
        df_panel,
        windows=ROLLING_WINDOWS,
        add_std=include_rolling_std,
        add_minmax_28=include_minmax_28,
    )

    if include_price_features:
        if df_prices is None:
            raise ValueError("include_price_features=True requires df_prices.")
        df_panel = add_price_features(df_panel, df_prices=df_prices)

    if include_cluster_features:
        if df_clusters is None:
            raise ValueError("include_cluster_features=True requires df_clusters.")
        df_panel = add_cluster_features(df_panel, df_clusters=df_clusters)

    validate_model_panel(df_panel)
    return _reduce_panel_dtypes(df_panel)


# -----------------------------------------------------------------------------
# Optional helper for latest feature snapshot
# -----------------------------------------------------------------------------

def get_latest_snapshot(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Return the latest available row per store × item.
    Useful for inference prototypes or API demos.
    """
    df = (
        df_panel.sort_values([STORE_COL, ITEM_COL, DATE_COL])
        .groupby([STORE_COL, ITEM_COL], observed=True, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return df
