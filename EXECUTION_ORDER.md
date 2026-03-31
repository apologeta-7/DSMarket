# Orden de ejecución — DSMarket TFM

## Orden recomendado

| Paso | Notebook | Entorno | Tiempo estimado |
|------|----------|---------|-----------------|
| 1 | 00_TFM_DSMarket.ipynb | cualquiera | solo lectura |
| 2 | 01_EDA.ipynb | Colab (15GB RAM) | ~20 min |
| 3 | 02_clustering.ipynb | local (8GB RAM) | ~10 min |
| 4 | 03_forecasting.ipynb | Kaggle (RAM ×4) | ~90 min |
| 5 | 04_stock_replenishment.ipynb | cualquiera | solo lectura |
| 6 | 05_pipeline_api.ipynb | local (8GB RAM) | ~15 min |

## Requisitos previos

### Antes de ejecutar cualquier notebook
- Los tres CSV deben estar en `data/raw/`
- Instalar dependencias: `pip install -r requirements.txt`

### Antes de ejecutar 05_pipeline_api.ipynb
- Los artefactos del forecasting deben existir en `models/`
- Archivos necesarios:
  - `models/model_E2.txt`
  - `models/preds_E2_calibrado.npy`
  - `models/metrics_final_calibrado.csv`
  - `models/calibration_factor.json`

## Nota sobre entornos

El proyecto está estructurado en notebooks modulares ejecutados
en el entorno más adecuado según el coste computacional de cada tarea:

- **EDA**: Colab por acceso a Drive y RAM suficiente
- **Clustering**: local, dataset agregado manejable en 8GB
- **Forecasting**: Kaggle con RAM ×4, dataset de 58M filas
- **Pipeline y API**: local, trabaja solo con artefactos guardados

Esta separación es una decisión de arquitectura técnica deliberada,
no una limitación. Prioriza claridad, reproducibilidad
y separación de responsabilidades entre tareas.