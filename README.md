# <img src="https://img.shields.io/badge/🛒-DSMarket-2D8C4E?style=for-the-badge" alt="DSMarket"/>

<div align="center">

# DSMarket
## Trabajo Final de Máster — Data Science & AI
**Nuclio Digital School · 2025**

*Forecasting de demanda, propuesta de reposición de stock*
*y planteamiento de productivización para una cadena de supermercados*

---

[![Python](https://img.shields.io/badge/Python-3.8+-2D8C4E?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-2D8C4E?style=flat-square)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F4A261?style=flat-square)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3-2D8C4E?style=flat-square)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-F4A261?style=flat-square)](LICENSE)

</div>

---

## 📋 Índice

- [Autores](#-autores)
- [Descripción del proyecto](#-descripción-del-proyecto)
- [Preguntas clave](#-preguntas-clave)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Guía de lectura de los notebooks](#-guía-de-lectura-de-los-notebooks)
- [Dataset y fuentes de datos](#-dataset-y-fuentes-de-datos)
- [Instalación y configuración](#-instalación-y-configuración)
- [Tecnologías utilizadas](#-tecnologías-utilizadas)
- [Resultados principales](#-resultados-principales)
- [Licencia](#-licencia)

---

## 👤 Autor

**Mateo Pascual Esseiva**  
📍 Spain · Open to remote  
💼 Data Analyst | MSc Data Science & AI (Nuclio Digital School)  
📧 [m.pascual.ess@gmail.com](mailto:m.pascual.ess@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/mateopascualesseiva/)  
🐙 [GitHub](https://github.com/apologeta-7)

**Tutor:** Raquel Revilla Bous — Nuclio Digital School

---

## 🎯 Descripción del proyecto

**DSMarket** (anteriormente TradiStores) es una pequeña cadena de supermercados en Estados Unidos que se encuentra en una fase temprana de transformación digital. Opera en **3 ciudades** (Nueva York, Boston y Philadelphia) con **10 tiendas activas** y **3 categorías de producto**: SUPERMARKET, HOME & GARDEN y ACCESSORIES.

Este TFM aborda un caso de uso de **role-play**: el equipo asume el papel de Nicole Chen, Data Scientist Sénior, encargada de impulsar iniciativas analíticas para el área financiera y operativa.

El proyecto cubre el ciclo completo desde la exploración de datos hasta una propuesta de productivización:

```
EDA y negocio → Segmentación → Forecasting → Reposición de stock → Pipeline y API
```

---

## ❓ Preguntas clave

El proyecto se articula alrededor de cuatro preguntas planteadas por la dirección de DSMarket:

> **1.** ¿Sigue siendo válido predecir a nivel tienda × producto y agregar después a niveles superiores?
>
> **2.** ¿Cómo construir un forecasting de demanda con horizonte de 28 días de forma razonable y defendible?
>
> **3.** ¿Cómo conectar ese forecasting con una lógica operativa de reposición de stock?
>
> **4.** ¿Cómo plantear una vía realista de productivización mediante código modular y una API?

---

## 📁 Estructura del repositorio

```
DSMarket_TFM/
│
├── 📓 notebook/
│   ├── 00_tfm_dsmarket.ipynb          # Portada, índice y guía de lectura
│   ├── 01_eda-business.ipynb          # Análisis exploratorio e historia de negocio
│   ├── 02_clustering.ipynb            # Segmentación de productos y tiendas
│   ├── 03_forecasting-final.ipynb     # Construcción y evaluación del modelo
│   ├── 04_stock.ipynb                 # Propuesta de reposición de stock
│   └── 05_pipeline_api.ipynb          # Pipeline modular y diseño de API
│
├── 📦 src/                            # Código modular reutilizable
│   ├── __init__.py
│   ├── config.py                      # Parámetros globales del proyecto
│   ├── paths.py                       # Rutas portables entre entornos
│   ├── preprocessing.py               # Limpieza y transformación de datos
│   ├── preprocessing_suggested.py     # Versión alternativa de preprocesado
│   ├── forecasting_utils.py           # Utilidades del modelo de forecasting
│   ├── plotting.py                    # Visualizaciones reutilizables
│   ├── utils.py                       # Funciones de apoyo generales
│   └── api/                           # Endpoint FastAPI
│       ├── __init__.py
│       ├── main.py                    # Aplicación FastAPI
│       ├── schemas.py                 # Modelos Pydantic (request/response)
│       └── service.py                 # Lógica de inferencia
│
├── 📂 data/
│   ├── raw/                           # Datos originales (solo lectura)
│   ├── processed/                     # Datos listos para modelado
│   │   ├── clusters_productos.csv
│   │   └── clusters_tiendas.csv
│   └── csv para power bi/             # Exportaciones para el dashboard BI
│       ├── clusters_productos.csv
│       ├── clusters_tiendas.csv
│       ├── dim_category.csv
│       ├── dim_city.csv
│       ├── dim_week.csv
│       ├── fact_city_cat_week.csv
│       ├── fact_city_week.csv
│       ├── fact_forecast_next_week_demo.csv
│       └── profile_priority.csv
│
├── 🤖 models/
│   ├── model_E1.txt                   # Modelo LightGBM E1 (lags + rolling + calendario)
│   ├── model_E2.txt                   # Modelo LightGBM E2 (+ precio + eventos)
│   ├── model_E3.txt                   # Modelo LightGBM E3 (+ cluster de producto)
│   ├── calibration_factor.json        # Factor de calibración global (×1.3657)
│   ├── metrics_final.csv              # Métricas comparativas todos los modelos
│   ├── metrics_final_calibrado.csv    # Métricas del modelo E2 calibrado en test
│   ├── preds_E1.npy                   # Predicciones brutas E1
│   ├── preds_E2.npy                   # Predicciones brutas E2
│   ├── preds_E2_calibrado.npy         # Predicciones E2 calibradas
│   ├── preds_E3.npy                   # Predicciones brutas E3
│   ├── weekly_calibrado.csv           # Agregación semanal calibrada
│   ├── y_true_test.npy                # Valores reales del período de test
│   └── idx_test.npy                   # Índices del split de test
│
├── 📊 outputs/                        # Exportables y resultados finales
│   ├── g1_serie_temporal.png
│   ├── g2_patron_semanal.png
│   ├── g3_heatmap.png
│   ├── g4_codo_silueta.png
│   ├── g5_scatter_clusters.png
│   ├── g6_pred_vs_real.png
│   ├── g7_sesgo_calibracion.png
│   ├── DSMarket Forecast API - Swagger UI.pdf
│   ├── DSMarket Forecast API (2) - Swagger UI.pdf
│   └── powerbi/
│       └── TFM mateo v01.pbix         # Dashboard Power BI
│
├── 📄 reports/                        # CSVs de apoyo para informes
│   ├── dim_category.csv
│   ├── dim_city.csv
│   ├── dim_week.csv
│   ├── fact_city_week.csv
│   └── profile_priority.csv
│
├── EXECUTION_ORDER.md                 # Guía de ejecución y entornos recomendados
├── requirements.txt
├── LICENSE
└── README.md
```

> ⚠️ Los archivos CSV del dataset original no están incluidos en el repositorio por su tamaño.
> Ver la sección [Dataset y fuentes de datos](#-dataset-y-fuentes-de-datos) para obtenerlos.

---

## 📖 Guía de lectura de los notebooks

La secuencia recomendada sigue esta lógica:

| # | Notebook | Propósito | Entorno recomendado |
|---|----------|-----------|---------------------|
| 00 | `00_tfm_dsmarket.ipynb` | Portada, índice y contexto del proyecto | Cualquiera |
| 01 | `01_eda-business.ipynb` | Análisis exploratorio y lectura de negocio | Colab / Kaggle |
| 02 | `02_clustering.ipynb` | Segmentación de productos (K-Means, K=4) | Local / Kaggle |
| 03 | `03_forecasting-final.ipynb` | Forecasting LightGBM — núcleo técnico | Kaggle (RAM ampliada) |
| 04 | `04_stock.ipynb` | Propuesta operativa de reposición | Cualquiera |
| 05 | `05_pipeline_api.ipynb` | Pipeline modular y diseño de API | Local |

> 💡 El notebook 03 requiere memoria ampliada en Kaggle por el tamaño del panel tienda × producto × día.

---

## 🗄️ Dataset y fuentes de datos

El proyecto utiliza el dataset público **M5 Forecasting — Accuracy** (Kaggle, 2020), reinterpretado como el histórico de ventas de DSMarket.

| Archivo | Descripción |
|---------|-------------|
| `item_sales.csv` | Ventas diarias por tienda × producto (formato ancho, d_1…d_N) |
| `item_prices.csv` | Precio semanal por tienda × producto |
| `daily_calendar_with_events.csv` | Calendario con eventos especiales |

**Fuente:** [M5 Forecasting — Accuracy · Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

Para reproducir el proyecto, descarga los tres archivos y colócalos en `data/raw/`.

---

## 🔧 Instalación y configuración

### Requisitos previos

- Python 3.8 o superior
- pip actualizado

### Instalación

```bash
# 1. Clonar o descargar el repositorio
git clone <url-del-repositorio>
cd DSMarket_TFM

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Activar en macOS/Linux:
source venv/bin/activate

# Activar en Windows:
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Estructura de datos

```bash
# Coloca los archivos del dataset en:
data/raw/item_sales.csv
data/raw/item_prices.csv
data/raw/daily_calendar_with_events.csv
```

### Cargar el modelo directamente

Si solo quieres reproducir la inferencia sin reentrenar:

```python
import lightgbm as lgb
import json

# Cargar modelo y factor de calibración
model = lgb.Booster(model_file="models/model_E2.txt")
with open("models/calibration_factor.json", "r") as f:
    calib_factor = json.load(f)  # 1.3657

# Aplicar predicción calibrada
pred_raw = model.predict(X)
pred_calibrado = pred_raw * calib_factor
```

---

## 🛠️ Tecnologías utilizadas

| Categoría | Tecnología |
|-----------|-----------|
| **Lenguaje** | Python 3.8+ |
| **Datos** | Pandas · NumPy · PyArrow |
| **ML / Forecasting** | LightGBM · scikit-learn · statsmodels |
| **Clustering** | K-Means (scikit-learn) |
| **Visualización** | Matplotlib · Seaborn · Plotly |
| **BI / Dashboard** | Power BI |
| **API** | FastAPI (diseño y demostración) |
| **Entorno** | Kaggle · Google Colab · Local |

---

## 📈 Resultados principales

### Clustering de productos (K=4)

| Cluster | Nombre | % catálogo | Características |
|---------|--------|-----------|-----------------|
| 1 | Básicos de alto volumen | 47% | Alta rotación, precio ~$3.59, baja volatilidad |
| 3 | Nicho premium | 21% | Precio ~$11, baja rotación, ticket alto |
| 0 | Estacionales volátiles | 19% | Alta variabilidad, patrón estacional fuerte |
| 2 | Baja rotación volátil | 13% | Comportamiento poco consolidado |

### Forecasting — Modelo ganador: E2 Calibrado

| Modelo | Descripción | Resultado |
|--------|-------------|-----------|
| E0a / E0b | Baselines (media histórica / media móvil) | Referencia mínima |
| E1 | LightGBM + lags + rolling + calendario | Mejora vs baseline |
| **E2** | **LightGBM + exógenas (precio + eventos)** | **✅ Mejor técnico** |
| E2 calibrado | E2 × factor 1.3657 | **✅ Solución operativa recomendada** |
| E3 | E2 + cluster de producto | Sin mejora adicional sobre E2 |

> El modelo E2 mostró infra-predicción sistemática en test. La calibración global (×1.3657) corrige el sesgo y lo convierte en la solución más adecuada para reposición de stock.

### Fórmula de reposición propuesta

```
Pedido_recomendado = max(0, Demanda_calibrada_7d + Safety_stock − Stock_disponible)

Safety_stock = z × σ_error × √L
```

Donde `z` depende del nivel de servicio objetivo: 90% · 95% · 99%

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

<div align="center">

**Nuclio Digital School — Máster en Data Science & AI**

*Trabajo Final de Máster — DSMarket — 2026*

*Mateo Pascual Esseiva*

</div>
