from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "momentum_12m",
    "momentum_6m",
    "momentum_3m",
    "volatility_3m",
    "volatility_1m",
    "size",
    "value",
    "profitability",
    "growth",
    "quality",
    "earnings_yield",
    "leverage",
    "liquidity",
]


def available_features(df: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in DEFAULT_FEATURES:
        if col in df.columns and df[col].notna().any():
            feature_cols.append(col)
    return feature_cols


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.sort_values(["ticker", "date"])
    df["momentum_12m"] = df.groupby("ticker")["close"].pct_change(252)
    df["momentum_6m"] = df.groupby("ticker")["close"].pct_change(126)
    df["momentum_3m"] = df.groupby("ticker")["close"].pct_change(63)
    df["volatility_3m"] = (
        df.groupby("ticker")["return"].rolling(63).std().reset_index(level=0, drop=True)
    )
    df["volatility_1m"] = (
        df.groupby("ticker")["return"].rolling(21).std().reset_index(level=0, drop=True)
    )

    market_cap = df.get("market_cap")
    if market_cap is None or market_cap.isna().all():
        shares = df.get("shares_outstanding", 1)
        df["market_cap"] = df["close"] * shares

    df["size"] = np.log(df["market_cap"].replace(0, np.nan))
    df["value"] = df.get("book_value", np.nan) / df.get("market_cap", np.nan)
    df["profitability"] = df.get("net_income", np.nan) / df.get("total_assets", np.nan)
    revenue = df.get("revenue")
    if revenue is not None and revenue.notna().any():
        df["quality"] = df.get("net_income", np.nan) / revenue
    else:
        df["quality"] = np.nan

    pe_ratio = df.get("pe_ratio")
    if pe_ratio is not None and pe_ratio.notna().any():
        df["earnings_yield"] = 1 / pe_ratio.replace(0, np.nan)
    else:
        df["earnings_yield"] = np.nan

    df["leverage"] = df.get("total_assets", np.nan) / df.get("market_cap", np.nan)

    volume = df.get("volume")
    if volume is not None and volume.notna().any():
        rolling_volume = (
            df.groupby("ticker")["volume"].rolling(21).mean().reset_index(level=0, drop=True)
        )
        df["liquidity"] = np.log1p(rolling_volume)
    else:
        df["liquidity"] = np.nan

    if "revenue" in df.columns:
        df["growth"] = df.groupby("ticker")["revenue"].pct_change(4)
    else:
        df["growth"] = np.nan

    df["return_next"] = df.groupby("ticker")["return"].shift(-1)

    feature_cols = available_features(df)
    df = df.dropna(subset=feature_cols + ["return_next"]).reset_index(drop=True)

    return df
