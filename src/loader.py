"""
Data loaders for:
  prices       → jacksoncrow/stock-market-dataset  (data/raw/prices/stocks/<TICKER>.csv)
  fundamentals → cnic92/200-financial-indicators   (data/raw/fundamentals/<YYYY>_Financial_Data.csv)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import FUNDAMENTALS_DIR, PRICES_DIR
from src.helpers import build_column_map, list_csv_files, normalize_columns, read_csv

log = logging.getLogger(__name__)

_PRICE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "date":   ("date", "timestamp", "datetime"),
    "open":   ("open", "open_price"),
    "high":   ("high", "high_price"),
    "low":    ("low", "low_price"),
    "close":  ("adj_close", "close", "close_price"),   # prefer adjusted
    "volume": ("volume", "vol"),
}

# cnic92 column names after normalize_col():
#   "Price/Earnings"       → price_earnings
#   "Book Value per Share" → book_value_per_share
#   "Market Cap"           → market_cap
#   "Unnamed: 0"           → unnamed_0   (= ticker index)
_FUND_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ticker":             ("unnamed_0", "ticker", "symbol", "tic"),
    "market_cap":         ("market_cap", "mktcap"),
    "book_value":         ("book_value_per_share", "book_value"),
    "pe_ratio":           ("price_earnings", "pe_ratio", "pe", "pe_ttm"),
    "pb_ratio":           ("price_book", "pb_ratio", "pb"),
    "revenue":            ("revenue", "total_revenue", "sales"),
    "net_income":         ("net_income", "netincome"),
    "total_assets":       ("total_assets", "assets"),
    "shares_outstanding": ("shares_outstanding", "shares_out"),
    "eps":                ("eps_diluted", "eps"),
    "roe":                ("return_on_equity", "roe"),
    "roa":                ("return_on_assets", "roa"),
    "debt_equity":        ("debt_equity", "debt_to_equity"),
    "revenue_growth":     ("revenue_growth",),
    "ebitda":             ("ebitda",),
    "free_cash_flow":     ("free_cash_flow",),
    "operating_income":   ("operating_income",),
}


def _load_price_file(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    df = df.rename(columns=normalize_columns(df.columns))
    col_map = build_column_map(list(df.columns), _PRICE_CANDIDATES)
    df = df.rename(columns={v: k for k, v in col_map.items() if v and v != k})
    if "ticker" not in df.columns:
        df["ticker"] = path.stem.upper()
    required = {"date", "ticker", "close"}
    if missing := required - set(df.columns):
        log.warning("Skipping %s — missing: %s", path.name, missing)
        return pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    keep = [c for c in ("date", "ticker", "open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep]


def load_prices(path: Path | None = None, max_tickers: int = 200) -> pd.DataFrame:
    if path is not None:
        return _load_price_file(path)
    folder = PRICES_DIR / "stocks" if (PRICES_DIR / "stocks").exists() else PRICES_DIR
    files  = list_csv_files(folder)[:max_tickers]
    if not files:
        raise FileNotFoundError(
            f"No price CSVs in {folder}.\n"
            "  kaggle datasets download -d jacksoncrow/stock-market-dataset -p data/raw/prices/ --unzip"
        )
    frames = [f for f in (_load_price_file(p) for p in files) if not f.empty]
    if not frames:
        raise ValueError("All price files were empty or invalid.")
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result.dropna(subset=["date", "close"]).reset_index(drop=True)


def _infer_year(path: Path) -> int | None:
    for part in path.stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def _load_fund_file(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    df = df.rename(columns=normalize_columns(df.columns))
    col_map = build_column_map(list(df.columns), _FUND_CANDIDATES)
    df = df.rename(columns={v: k for k, v in col_map.items() if v and v != k})
    if "ticker" not in df.columns:
        log.warning("Skipping %s — no ticker column.", path.name)
        return pd.DataFrame()
    if "date" not in df.columns:
        year = _infer_year(path)
        if not year:
            return pd.DataFrame()
        df["date"] = pd.Timestamp(year=year, month=12, day=31)
    df = df.drop(columns=[c for c in df.columns if c in {"class", "y"}], errors="ignore")
    return df.loc[:, ~df.columns.duplicated()]


def load_fundamentals(path: Path | None = None) -> pd.DataFrame:
    if path is not None:
        return _load_fund_file(path)
    files = list_csv_files(FUNDAMENTALS_DIR)
    if not files:
        raise FileNotFoundError(
            f"No fundamentals CSVs in {FUNDAMENTALS_DIR}.\n"
            "  kaggle datasets download -d cnic92/200-financial-indicators -p data/raw/fundamentals/ --unzip"
        )
    frames = [f for f in (_load_fund_file(p) for p in files) if not f.empty]
    if not frames:
        raise ValueError("All fundamentals files were empty or invalid.")
    result = pd.concat(frames, ignore_index=True)
    result["date"]   = pd.to_datetime(result["date"], errors="coerce")
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    return result.dropna(subset=["date", "ticker"]).reset_index(drop=True)
