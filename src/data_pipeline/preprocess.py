from __future__ import annotations

import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["close"].pct_change()
    return prices


def merge_price_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    fundamentals = fundamentals.copy()

    prices["date"] = pd.to_datetime(prices["date"])
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])

    prices = prices.sort_values(["ticker", "date"])
    fundamentals = fundamentals.sort_values(["ticker", "date"])

    merged = pd.merge_asof(
        prices.sort_values("date"),
        fundamentals.sort_values("date"),
        on="date",
        by="ticker",
        direction="backward",
    )

    merged = merged.dropna(subset=["return"])
    return merged
