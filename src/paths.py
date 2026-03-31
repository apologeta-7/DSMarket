# =============================================================================
# src/paths.py
# Rutas del proyecto — independientes del entorno y del usuario
# =============================================================================

from pathlib import Path

# Raíz del proyecto: sube dos niveles desde src/
ROOT = Path(__file__).resolve().parents[1]

# Datos
DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
INTERIM_DIR    = DATA_DIR / "interim"
PROCESSED_DIR  = DATA_DIR / "processed"

# Modelos y artefactos
MODELS_DIR     = ROOT / "models"

# Outputs
OUTPUTS_DIR    = ROOT / "outputs"
FIGURES_DIR    = OUTPUTS_DIR / "figures"
TABLES_DIR     = OUTPUTS_DIR / "tables"
REPORTS_DIR    = OUTPUTS_DIR / "reports"

# CSV de Power BI (output clustering)
PBI_DIR        = ROOT / "csv para power bi"

# Archivos de datos raw
SALES_PATH     = RAW_DIR / "item_sales.csv"
CALENDAR_PATH  = RAW_DIR / "daily_calendar_with_events.csv"
PRICES_PATH    = RAW_DIR / "item_prices.csv"

# Artefactos del forecasting
MODEL_E2_PATH          = MODELS_DIR / "model_E2.txt"
PREDS_E2_CAL_PATH      = MODELS_DIR / "preds_E2_calibrado.npy"
Y_TRUE_PATH            = MODELS_DIR / "y_true_test.npy"
IDX_TEST_PATH          = MODELS_DIR / "idx_test.npy"
METRICS_PATH           = MODELS_DIR / "metrics_final_calibrado.csv"
WEEKLY_CAL_PATH        = MODELS_DIR / "weekly_calibrado.csv"
CLUSTERS_PATH          = PBI_DIR   / "clusters_productos.csv"

# Crear directorios si no existen (útil en entornos nuevos)
for _dir in [INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("✅ Rutas del proyecto:")
    print(f"   ROOT:         {ROOT}")
    print(f"   RAW_DIR:      {RAW_DIR}")
    print(f"   MODELS_DIR:   {MODELS_DIR}")
    print(f"   OUTPUTS_DIR:  {OUTPUTS_DIR}")
    print(f"   RAW existe:   {RAW_DIR.exists()}")
    print(f"   MODELS existe:{MODELS_DIR.exists()}")
