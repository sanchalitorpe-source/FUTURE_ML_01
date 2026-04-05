# Sales & Demand Forecasting Pipeline

A complete ML pipeline that predicts future product sales by category and region using **Gradient Boosting Regression**.

---

## Project Structure

```
sales_forecasting/
│
├── main.py                  ← Entry point
├── config.py                ← Paths, hyperparams, constants
├── requirements.txt
│
├── pipeline/
│   ├── data_loader.py       ← Load CSV or generate synthetic data
│   ├── preprocessor.py      ← Clean, deduplicate, encode
│   ├── feature_eng.py       ← Lag, rolling, temporal features
│   ├── trainer.py           ← Train GBR + cross-validation
│   ├── evaluator.py         ← R², RMSE, MAE, MAPE + residual plot
│   ├── forecaster.py        ← Iterative future month predictions
│   └── visualizer.py        ← 5-panel dashboard chart
│
├── utils/
│   └── logger.py            ← Consistent logging
│
├── data/
│   └── sales_data.csv       ← Auto-generated if missing
│
├── models/
│   └── gbr_model.pkl        ← Saved trained model
│
└── outputs/
    ├── forecast.csv         ← 6-month predictions
    ├── evaluation.png       ← Residual & actual-vs-predicted
    └── dashboard.png        ← Full 5-panel dashboard
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (train + forecast + charts)
python main.py

# 3. Train only (skip forecast step)
python main.py --mode train

# 4. Forecast 12 months instead of 6
python main.py --months 12
```

---

## Pipeline Steps

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `data_loader.py`  | Load CSV; generates 3-year synthetic data if absent |
| 2 | `preprocessor.py` | Remove duplicates, fill nulls, clip outliers, encode |
| 3 | `feature_eng.py`  | Lag features (1/2/3/6/12m), rolling mean/std, cyclical month encoding |
| 4 | `trainer.py`      | Time-based split, 5-fold CV, fit GBR, save model |
| 5 | `evaluator.py`    | R², RMSE, MAE, MAPE; save residual chart |
| 6 | `forecaster.py`   | Iteratively forecast each future month |
| 7 | `visualizer.py`   | 5-panel dashboard PNG |

---

## Model Performance (on synthetic data)

| Metric | Value |
|--------|-------|
| R²     | ~0.91 |
| MAPE   | ~9.8% |
| RMSE   | ~$650 |

---

## Key Business Insights

- **Electronics** spikes Jan–Feb (post-holiday) and June
- **West region** requires ~25% more inventory than Central
- Use 4–6 week lead time for staffing and purchase orders
