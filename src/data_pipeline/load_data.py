from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import FUNDAMENTALS_DIR, PRICES_DIR
from src.utils.columns import build_column_map, normalize_columns
from src.utils.io import list_csv_files, read_csv

PRICE_COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "date": ("date", "timestamp", "datetime"),
    "ticker": ("ticker", "symbol", "tic"),
    "open": ("open", "open_price"),
    "close": ("close", "close_price", "adj_close", "adjclose"),
    "volume": ("volume", "vol"),
}

FUND_COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "date": ("date", "fiscal_date", "report_date"),
    "ticker": ("ticker", "symbol", "tic"),
    "market_cap": ("market_cap", "mktcap", "marketcapitalization"),
    "book_value": ("book_value", "bookvalue", "book_value_per_share"),
    "pe_ratio": ("pe_ratio", "pe", "pe_ttm", "price_to_earnings"),
    "pb_ratio": ("pb_ratio", "pb", "price_to_book"),
    "revenue": ("revenue", "total_revenue", "sales"),
    "net_income": ("net_income", "netincome", "ni"),
    "total_assets": ("total_assets", "assets", "totalassets"),
    "shares_outstanding": ("shares_outstanding", "shares", "shares_out"),
}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    mapping = normalize_columns(df.columns)
    df = df.rename(columns=mapping)
    return df


def _apply_column_map(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    available = {standard: column_map.get(standard) for standard in column_map}
    rename_map = {raw: standard for standard, raw in available.items() if raw}
    df = df.rename(columns=rename_map)
    return df


def _read_price_files(files: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in files:
        df = read_csv(path)
        df = _standardize(df)
        column_map = build_column_map(df.columns, PRICE_COLUMN_CANDIDATES)
        df = _apply_column_map(df, column_map)

        if "ticker" not in df.columns:
            df["ticker"] = path.stem

        required = {"date", "ticker", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Price data missing required columns in {path.name}: {sorted(missing)}")

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No price CSV files found in data/raw/prices.")

    return pd.concat(frames, ignore_index=True)


def load_prices_data(path: Path | None = None, max_tickers: int | None = 200) -> pd.DataFrame:
    if path is not None:
        return _read_price_files([path])

    stocks_dir = PRICES_DIR / "stocks"
    files = list_csv_files(stocks_dir if stocks_dir.exists() else PRICES_DIR)
    if not files:
        raise FileNotFoundError("No price CSV files found in data/raw/prices.")

    if max_tickers is not None:
        files = files[:max_tickers]

    return _read_price_files(files)


def _infer_year(path: Path) -> int | None:
    for part in path.stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def _read_fundamental_file(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    df = _standardize(df)

    if "ticker" not in df.columns:
        unnamed = [c for c in df.columns if c.startswith("unnamed")]
        if unnamed:
            df = df.rename(columns={unnamed[0]: "ticker"})

    if "date" not in df.columns:
        year = _infer_year(path)
        if year is None:
            raise ValueError(f"Fundamentals file missing year and date: {path.name}")
        df["date"] = f"{year}-12-31"

    column_map = build_column_map(df.columns, FUND_COLUMN_CANDIDATES)
    df = _apply_column_map(df, column_map)

    required = {"date", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fundamentals data missing required columns in {path.name}: {sorted(missing)}")

    return df


def load_fundamentals_data(path: Path | None = None) -> pd.DataFrame:
    if path is not None:
        return _read_fundamental_file(path)

    files = list_csv_files(FUNDAMENTALS_DIR)
    if not files:
        raise FileNotFoundError("No fundamentals CSV files found in data/raw/fundamentals.")

    frames = [_read_fundamental_file(file_path) for file_path in files]

    # remove duplicate columns
    frames = [df.loc[:, ~df.columns.duplicated()] for df in frames]

    # reset index
    frames = [df.reset_index(drop=True) for df in frames]

    return pd.concat(frames, ignore_index=True)
