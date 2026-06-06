from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorize(series: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return series.clip(series.quantile(lo), series.quantile(hi))


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = (prices
              .sort_values(["ticker", "date"])
              .drop_duplicates(subset=["ticker", "date"], keep="last")
              .reset_index(drop=True))
    prices["return"] = prices.groupby("ticker")["close"].pct_change()
    mask = prices["return"].notna()
    prices.loc[mask, "return"] = _winsorize(prices.loc[mask, "return"])
    return prices


_FUND_KEEP = [
    "ticker", "date",
    "book_value", "net_income", "total_assets", "revenue",
    "pe_ratio", "roe", "roa", "shares_outstanding", "market_cap",
]


def merge_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy().sort_values(["ticker", "date"])

    keep = [c for c in _FUND_KEEP if c in fundamentals.columns]
    fundamentals = fundamentals[keep].copy().sort_values(["ticker", "date"])

    prices["date"]       = pd.to_datetime(prices["date"]).dt.as_unit("ns")
    fundamentals["date"] = pd.to_datetime(fundamentals["date"]).dt.as_unit("ns")

    skip = {"date", "ticker"}
    for col in fundamentals.columns:
        if col not in skip:
            fundamentals[col] = pd.to_numeric(fundamentals[col], errors="coerce")

    merged = pd.merge_asof(
        prices.sort_values("date"),
        fundamentals.sort_values("date"),
        on="date", by="ticker", direction="backward",
    )
    return merged.dropna(subset=["return"]).reset_index(drop=True)
