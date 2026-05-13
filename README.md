# <img src="https://img.shields.io/badge/🛒-DSMarket-2D8C4E?style=for-the-badge" alt="DSMarket"/>

<div align="center">

# DSMarket
## Master's Final Project — Data Science & AI
**Nuclio Digital School · 2025**

*Demand forecasting, stock replenishment proposal*
*and productization approach for a supermarket chain*

---

[![Python](https://img.shields.io/badge/Python-3.8+-2D8C4E?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-2D8C4E?style=flat-square)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F4A261?style=flat-square)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3-2D8C4E?style=flat-square)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-F4A261?style=flat-square)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Author](#-author)
- [Project description](#-project-description)
- [Key questions](#-key-questions)
- [Repository structure](#-repository-structure)
- [Notebook reading guide](#-notebook-reading-guide)
- [Dataset and data sources](#-dataset-and-data-sources)
- [Installation and setup](#-installation-and-setup)
- [Technologies used](#-technologies-used)
- [Main results](#-main-results)
- [License](#-license)

---

## 👤 Author

**Mateo Pascual Esseiva**  
📍 Spain · Open to remote  
💼 Data Analyst | MSc Data Science & AI (Nuclio Digital School)  


---

## 🎯 Project description

**DSMarket** (formerly TradiStores) is a small supermarket chain in the United States that is in an early stage of digital transformation. It operates in **3 cities** (New York, Boston and Philadelphia) with **10 active stores** and **3 product categories**: SUPERMARKET, HOME & GARDEN and ACCESSORIES.

This Master's Final Project addresses a **role-play** use case: the team assumes the role of Nicole Chen, Senior Data Scientist, responsible for driving analytics initiatives for the finance and operations areas.

The project covers the full cycle from data exploration to a productization proposal:

```text
EDA and business → Segmentation → Forecasting → Stock replenishment → Pipeline and API
```

---

## ❓ Key questions

The project is structured around four questions raised by DSMarket's leadership:

> **1.** Is it still valid to forecast at store × product level and then aggregate to higher levels?
>
> **2.** How can we build a reasonable and defensible 28-day demand forecast?
>
> **3.** How can that forecast be connected to an operational stock replenishment logic?
>
> **4.** How can we define a realistic productization path through modular code and an API?

---

## 📁 Repository structure

```text
DSMarket_TFM/
│
├── 📓 notebook/
│   ├── 00_tfm_dsmarket.ipynb          # Cover, index and reading guide
│   ├── 01_eda-business.ipynb          # Exploratory analysis and business story
│   ├── 02_clustering.ipynb            # Product and store segmentation
│   ├── 03_forecasting-final.ipynb     # Model construction and evaluation
│   ├── 04_stock.ipynb                 # Stock replenishment proposal
│   └── 05_pipeline_api.ipynb          # Modular pipeline and API design
│
├── 📦 src/                            # Reusable modular code
│   ├── __init__.py
│   ├── config.py                      # Global project parameters
│   ├── paths.py                       # Portable paths across environments
│   ├── preprocessing.py               # Data cleaning and transformation
│   ├── preprocessing_suggested.py     # Alternative preprocessing version
│   ├── forecasting_utils.py           # Forecasting model utilities
│   ├── plotting.py                    # Reusable visualizations
│   ├── utils.py                       # General helper functions
│   └── api/                           # FastAPI endpoint
│       ├── __init__.py
│       ├── main.py                    # FastAPI application
│       ├── schemas.py                 # Pydantic models (request/response)
│       └── service.py                 # Inference logic
│
├── 📂 data/
│   ├── raw/                           # Original data (read-only)
│   ├── processed/                     # Data ready for modeling
│   │   ├── clusters_productos.csv
│   │   └── clusters_tiendas.csv
│   └── csv para power bi/             # Exports for the BI dashboard
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
│   ├── model_E1.txt                   # LightGBM E1 model (lags + rolling + calendar)
│   ├── model_E2.txt                   # LightGBM E2 model (+ price + events)
│   ├── model_E3.txt                   # LightGBM E3 model (+ product cluster)
│   ├── calibration_factor.json        # Global calibration factor (×1.3657)
│   ├── metrics_final.csv              # Comparative metrics for all models
│   ├── metrics_final_calibrado.csv    # Calibrated E2 metrics on test
│   ├── preds_E1.npy                   # Raw E1 predictions
│   ├── preds_E2.npy                   # Raw E2 predictions
│   ├── preds_E2_calibrado.npy         # Calibrated E2 predictions
│   ├── preds_E3.npy                   # Raw E3 predictions
│   ├── weekly_calibrado.csv           # Calibrated weekly aggregation
│   ├── y_true_test.npy                # Actual values for the test period
│   └── idx_test.npy                   # Test split indices
│
├── 📊 outputs/                        # Final exports and results
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
│       └── TFM mateo v01.pbix         # Power BI dashboard
│
├── 📄 reports/                        # Supporting CSVs for reports
│   ├── dim_category.csv
│   ├── dim_city.csv
│   ├── dim_week.csv
│   ├── fact_city_week.csv
│   └── profile_priority.csv
│
├── EXECUTION_ORDER.md                 # Execution guide and recommended environments
├── requirements.txt
├── LICENSE
└── README.md
```

> ⚠️ The original dataset CSV files are not included in the repository because of their size.
> See the [Dataset and data sources](#-dataset-and-data-sources) section to obtain them.

---

## 📖 Notebook reading guide

The recommended sequence follows this logic:

| # | Notebook | Purpose | Recommended environment |
|---|----------|---------|-------------------------|
| 00 | `00_tfm_dsmarket.ipynb` | Cover, index and project context | Any |
| 01 | `01_eda-business.ipynb` | Exploratory analysis and business interpretation | Colab / Kaggle |
| 02 | `02_clustering.ipynb` | Product segmentation (K-Means, K=4) | Local / Kaggle |
| 03 | `03_forecasting-final.ipynb` | LightGBM forecasting — technical core | Kaggle (extended RAM) |
| 04 | `04_stock.ipynb` | Operational replenishment proposal | Any |
| 05 | `05_pipeline_api.ipynb` | Modular pipeline and API design | Local |

> 💡 Notebook 03 requires extended memory in Kaggle due to the size of the store × product × day panel.

---

## 🗄️ Dataset and data sources

The project uses the public **M5 Forecasting — Accuracy** dataset (Kaggle, 2020), reinterpreted as DSMarket's sales history.

| File | Description |
|------|-------------|
| `item_sales.csv` | Daily sales by store × product (wide format, d_1…d_N) |
| `item_prices.csv` | Weekly price by store × product |
| `daily_calendar_with_events.csv` | Calendar with special events |

**Source:** [M5 Forecasting — Accuracy · Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

To reproduce the project, download the three files and place them in `data/raw/`.

---

## 🔧 Installation and setup

### Prerequisites

- Python 3.8 or higher
- Updated pip

### Installation

```bash
# 1. Clone or download the repository
git clone <repository-url>
cd DSMarket_TFM

# 2. Create a virtual environment (recommended)
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Data structure

```bash
# Place the dataset files in:
data/raw/item_sales.csv
data/raw/item_prices.csv
data/raw/daily_calendar_with_events.csv
```

### Load the model directly

If you only want to reproduce inference without retraining:

```python
import lightgbm as lgb
import json

# Load model and calibration factor
model = lgb.Booster(model_file="models/model_E2.txt")
with open("models/calibration_factor.json", "r") as f:
    calib_factor = json.load(f)  # 1.3657

# Apply calibrated prediction
pred_raw = model.predict(X)
pred_calibrado = pred_raw * calib_factor
```

---

## 🛠️ Technologies used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **Data** | Pandas · NumPy · PyArrow |
| **ML / Forecasting** | LightGBM · scikit-learn · statsmodels |
| **Clustering** | K-Means (scikit-learn) |
| **Visualization** | Matplotlib · Seaborn · Plotly |
| **BI / Dashboard** | Power BI |
| **API** | FastAPI (design and demo) |
| **Environment** | Kaggle · Google Colab · Local |

---

## 📈 Main results

### Product clustering (K=4)

| Cluster | Name | % of catalog | Characteristics |
|---------|------|--------------|-----------------|
| 1 | High-volume essentials | 47% | High turnover, price ~$3.59, low volatility |
| 3 | Premium niche | 21% | Price ~$11, low turnover, high ticket |
| 0 | Volatile seasonal items | 19% | High variability, strong seasonal pattern |
| 2 | Volatile low-turnover | 13% | Less consolidated behavior |

### Forecasting — Winning model: Calibrated E2

| Model | Description | Result |
|-------|-------------|--------|
| E0a / E0b | Baselines (historical mean / moving average) | Minimum benchmark |
| E1 | LightGBM + lags + rolling + calendar | Improvement vs baseline |
| **E2** | **LightGBM + exogenous features (price + events)** | **✅ Best technical model** |
| Calibrated E2 | E2 × factor 1.3657 | **✅ Recommended operational solution** |
| E3 | E2 + product cluster | No additional improvement over E2 |

> The E2 model showed systematic underforecasting on the test set. Global calibration (×1.3657) corrects the bias and makes it the most suitable solution for stock replenishment.

### Proposed replenishment formula

```text
Recommended_order = max(0, Calibrated_demand_7d + Safety_stock − Available_stock)

Safety_stock = z × σ_error × √L
```

Where `z` depends on the target service level: 90% · 95% · 99%

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Nuclio Digital School — Master's Degree in Data Science & AI**

*Master's Final Project — DSMarket — 2026*

*Mateo Pascual Esseiva*

</div>
