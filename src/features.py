from __future__ import annotations

import numpy as np
import pandas as pd

ALL_FEATURES = [
    "momentum_12m", "momentum_6m", "momentum_3m", "reversal_1m",
    "volatility_3m", "volatility_1m",
    "size", "value", "profitability", "quality", "roe", "roa",
    "growth", "earnings_yield", "leverage", "liquidity",
]


def available(df: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURES if c in df.columns and df[c].notna().any()]


def _get(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index, dtype=float)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _xscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return s * 0.0 if (std == 0 or np.isnan(std)) else (s - s.mean()) / std


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    gc = df.groupby("ticker")["close"]
    gr = df.groupby("ticker")["return"]

    df["momentum_12m"] = gc.pct_change(252) - gc.pct_change(21)
    df["momentum_6m"]  = gc.pct_change(126)
    df["momentum_3m"]  = gc.pct_change(63)
    df["reversal_1m"]  = gc.pct_change(21)
    df["volatility_3m"] = gr.transform(lambda x: x.rolling(63, min_periods=20).std())
    df["volatility_1m"] = gr.transform(lambda x: x.rolling(21, min_periods=10).std())

    if "market_cap" not in df.columns or df["market_cap"].isna().all():
        df["market_cap"] = df["close"] * _get(df, "shares_outstanding").fillna(1)
    mc = df["market_cap"].replace(0, np.nan)

    df["size"]           = np.log(mc.clip(lower=1))
    df["value"]          = _safe_div(_get(df, "book_value"), mc)
    df["profitability"]  = _safe_div(_get(df, "net_income"), _get(df, "total_assets"))
    df["quality"]        = _safe_div(_get(df, "net_income"), _get(df, "revenue"))
    df["roe"]            = _get(df, "roe")
    df["roa"]            = _get(df, "roa")
    df["earnings_yield"] = _safe_div(pd.Series(1.0, index=df.index), _get(df, "pe_ratio"))
    df["leverage"]       = _safe_div(_get(df, "total_assets"), mc)

    if "revenue" in df.columns and df["revenue"].notna().any():
        df["growth"] = df.groupby("ticker")["revenue"].pct_change(4, fill_method=None)
    else:
        df["growth"] = np.nan

    if "volume" in df.columns and df["volume"].notna().any():
        df["liquidity"] = df.groupby("ticker")["volume"].transform(
            lambda x: np.log1p(x.rolling(21, min_periods=5).mean())
        )
    else:
        df["liquidity"] = np.nan

    df["return_next"] = df.groupby("ticker")["return"].shift(-1)

    feat_cols = available(df)
    for col in feat_cols:
        df[col] = df.groupby("date")[col].transform(_xscore)

    return df.dropna(subset=feat_cols + ["return_next"]).reset_index(drop=True)
