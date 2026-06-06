# FactorLens

A quantitative equity factor-discovery platform. Ingests stock price and fundamental data, engineers 15 cross-sectional factors, trains ML models to predict returns, and validates everything with a long-short backtest — all inside a Streamlit dashboard.

---

## Features

- 15 factors: momentum (12m/6m/3m), reversal, volatility (3m/1m), size, value, profitability, quality, ROE, earnings yield, leverage, growth, liquidity
- 3 ML models: LASSO, Random Forest, XGBoost — compared on MSE, R², and Spearman IC
- Long-short portfolio construction with cumulative return, Sharpe, and drawdown
- Factor correlation heatmap and rolling IC charts
- Portfolio factor-exposure calculator (custom tickers and weights)
- Market regime monitor (risk-on / risk-off / mixed)

---

## Project Structure

```
FactorLens/
├── app.py                  # Streamlit dashboard (entry point)
├── src/
│   ├── config.py           # Path constants
│   ├── helpers.py          # Column normalisation, CSV utilities
│   ├── loader.py           # Kaggle dataset loaders
│   ├── preprocess.py       # Returns calculation, as-of merge
│   ├── features.py         # Factor engineering + z-scoring
│   ├── portfolio.py        # Long-short returns, IC, backtest
│   ├── exposure.py         # Portfolio factor-exposure calculator
│   ├── train.py            # Model training and evaluation
│   └── charts.py           # Plotly chart functions
├── data/
│   └── processed/
│       ├── stock_features.csv   # 266,323 rows — 264 tickers, 2015–2018
│       └── factor_returns.csv   # 1,317 dates × 15 factors
├── requirements.txt
└── FactorLens.txt          # Detailed project + financial terms documentation
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app.py
```

App loads directly from `data/processed/` — no raw data download needed.

---

## Data

Processed data (264 tickers, 2015–2018) is included in `data/processed/`.

If you want to regenerate from scratch using the original Kaggle datasets:

1. Download raw data:
   ```bash
   kaggle datasets download -d jacksoncrow/stock-market-dataset -p data/raw/prices/ --unzip
   kaggle datasets download -d cnic92/200-financial-indicators -p data/raw/fundamentals/ --unzip
   ```

2. Run the pipeline script:
   ```bash
   python -c "
   from src.loader import load_prices, load_fundamentals
   from src.preprocess import compute_returns, merge_fundamentals
   from src.features import build, available
   from src.portfolio import factor_returns
   from src.config import PROCESSED_FEATURES, PROCESSED_FACTORS
   prices = compute_returns(load_prices(max_tickers=500))
   merged = merge_fundamentals(prices, load_fundamentals())
   df = build(merged)
   fret = factor_returns(df, available(df))
   PROCESSED_FEATURES.parent.mkdir(parents=True, exist_ok=True)
   df.to_csv(PROCESSED_FEATURES, index=False)
   fret.to_csv(PROCESSED_FACTORS)
   "
   ```

---

## Tech Stack

Python · Streamlit · pandas · scikit-learn · XGBoost · Plotly · scipy

---

## Deployment

**Streamlit Community Cloud** — connect the repo, set entrypoint to `app.py`.

> `data/processed/` is committed so the app works immediately on deploy without any Kaggle credentials.
