# =============================================================================
# src/plotting.py
# Funciones de visualización reutilizables — DSMarket TFM
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Paleta corporativa DSMarket
COLORS = {
    "primary":   "#2D8C4E",
    "secondary": "#F4A261",
    "accent":    "#E76F51",
    "neutral":   "#8ECAE6",
    "dark":      "#264653",
}


def save_figure(fig: plt.Figure,
                path: Path,
                dpi: int = 150) -> None:
    """Guarda figura en disco creando el directorio si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Figura guardada: {path}")


# =============================================================================
# Series temporales
# =============================================================================

def plot_time_series(df: pd.DataFrame,
                     date_col: str,
                     value_col: str,
                     title: str = "",
                     save_path: Path = None) -> plt.Figure:
    """
    Serie temporal simple con estilo DSMarket.

    Args:
        df:         DataFrame con columnas date_col y value_col
        date_col:   nombre de la columna de fechas
        value_col:  nombre de la columna de valores
        title:      título del gráfico
        save_path:  si se indica, guarda la figura

    Returns:
        figura matplotlib
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[date_col], df[value_col],
            color=COLORS["primary"], linewidth=1.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(value_col)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Real vs Predicho
# =============================================================================

def plot_real_vs_pred(dates: pd.Series,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      title: str = "Real vs Predicho",
                      save_path: Path = None) -> plt.Figure:
    """
    Gráfico de líneas real vs predicho con área de error.

    Args:
        dates:     Serie de fechas
        y_true:    valores reales
        y_pred:    valores predichos
        title:     título del gráfico
        save_path: si se indica, guarda la figura
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, y_true,
            color=COLORS["primary"], linewidth=2, label="Real")
    ax.plot(dates, y_pred,
            color=COLORS["accent"], linewidth=2,
            linestyle="--", label="Predicho")
    ax.fill_between(dates, y_true, y_pred,
                    alpha=0.15, color=COLORS["accent"])

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Comparativa de métricas entre modelos
# =============================================================================

def plot_metrics_comparison(df_metrics: pd.DataFrame,
                             metric: str = "MAE",
                             save_path: Path = None) -> plt.Figure:
    """
    Barplot comparativo de una métrica entre experimentos.

    Args:
        df_metrics: DataFrame con columnas [model, MAE, WAPE, Bias, ...]
        metric:     métrica a visualizar
        save_path:  si se indica, guarda la figura
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [COLORS["accent"] if i == 0 else COLORS["primary"]
              for i in range(len(df_metrics))]

    ax.bar(df_metrics["model"], df_metrics[metric],
           color=colors, alpha=0.85)

    # Añadir valor encima de cada barra
    for i, val in enumerate(df_metrics[metric]):
        ax.text(i, val + 0.01, f"{val:.3f}",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_title(f"Comparativa {metric} — Todos los experimentos",
                 fontsize=13)
    ax.set_ylabel(metric)
    ax.set_xticklabels(df_metrics["model"], rotation=25, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Bias por categoría
# =============================================================================

def plot_bias_by_category(df_bias: pd.DataFrame,
                           save_path: Path = None) -> plt.Figure:
    """
    Barplot de bias (%) por categoría.

    Args:
        df_bias: DataFrame con columnas [category, bias_pct]
        save_path: si se indica, guarda la figura
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = [COLORS["accent"] if b < 0 else COLORS["primary"]
              for b in df_bias["bias_pct"]]

    ax.bar(df_bias["category"], df_bias["bias_pct"],
           color=colors, alpha=0.85)
    ax.axhline(0, color=COLORS["dark"], linewidth=1.5, linestyle="--")

    for i, row in df_bias.iterrows():
        ax.text(i, row["bias_pct"] - 1.5, f"{row['bias_pct']:.1f}%",
                ha="center", fontsize=10,
                color="white", fontweight="bold")

    ax.set_title("Bias por categoría — E2 (% infrapredicción)", fontsize=13)
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Bias (%)")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Split temporal
# =============================================================================

def plot_temporal_split(df_panel: pd.DataFrame,
                        train_end,
                        test_start,
                        val_windows: list = None,
                        save_path: Path = None) -> plt.Figure:
    """
    Visualiza el esquema de split temporal train/validación/test.

    Args:
        df_panel:    panel completo (necesita columna 'date')
        train_end:   fecha fin del train
        test_start:  fecha inicio del test
        val_windows: lista de tuplas (v_start, v_end) opcionales
        save_path:   si se indica, guarda la figura
    """
    min_date = df_panel["date"].min()
    fig, ax  = plt.subplots(figsize=(14, 3))

    # Train
    ax.barh(0, (train_end - min_date).days,
            left=0, height=0.4,
            color=COLORS["primary"], alpha=0.8, label="Train")

    # Ventanas de validación
    if val_windows:
        for i, (v_start, v_end) in enumerate(val_windows):
            left  = (v_start - min_date).days
            width = (v_end - v_start).days
            ax.barh(0, width, left=left, height=0.4,
                    color=COLORS["secondary"], alpha=0.9,
                    label="Validación rolling" if i == 0 else "")

    # Test
    left_test = (test_start - min_date).days
    ax.barh(0, 28, left=left_test, height=0.4,
            color=COLORS["accent"], alpha=0.9, label="Test (28 días)")

    ax.set_xlabel("Días desde inicio del histórico")
    ax.set_yticks([])
    ax.set_title(
        "Esquema de split temporal — Train / Validación / Test",
        fontsize=13
    )
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35, axis="x")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Distribución de demanda (zero inflation)
# =============================================================================

def plot_demand_distribution(series: pd.Series,
                              title: str = "Distribución de demanda",
                              save_path: Path = None) -> plt.Figure:
    """
    Histograma de demanda con y sin ceros lado a lado.

    Args:
        series:    Serie de unidades vendidas
        title:     título del gráfico
        save_path: si se indica, guarda la figura
    """
    pct_zeros = (series == 0).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(series, bins=50,
                 color=COLORS["primary"], alpha=0.8)
    axes[0].set_title(f"Con ceros ({pct_zeros:.1f}% son 0)", fontsize=12)
    axes[0].set_xlabel("Units")
    axes[0].set_ylabel("Frecuencia")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].hist(series[series > 0], bins=50,
                 color=COLORS["secondary"], alpha=0.8)
    axes[1].set_title("Sin ceros (distribución real)", fontsize=12)
    axes[1].set_xlabel("Units")
    axes[1].set_ylabel("Frecuencia")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig
```

---

