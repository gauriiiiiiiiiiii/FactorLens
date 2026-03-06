from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

PRICES_DIR = DATA_RAW / "prices"
FUNDAMENTALS_DIR = DATA_RAW / "fundamentals"

PROCESSED_FEATURES = DATA_PROCESSED / "stock_features.csv"
PROCESSED_FACTORS = DATA_PROCESSED / "factor_returns.csv"
