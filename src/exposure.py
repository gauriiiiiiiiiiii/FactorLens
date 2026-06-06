from __future__ import annotations

import warnings

import pandas as pd


def portfolio_exposure(
    df: pd.DataFrame,
    portfolio: dict[str, float],
    feature_cols: list[str],
) -> pd.Series:
    latest = df.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

    missing = [t for t in portfolio if t not in latest.index]
    if missing:
        warnings.warn(f"Tickers not in data (skipped): {missing}", stacklevel=2)

    exposures = {}
    for feat in feature_cols:
        if feat not in latest.columns:
            continue
        exposures[feat] = sum(
            w * float(latest.loc[t, feat])
            for t, w in portfolio.items()
            if t in latest.index and pd.notna(latest.loc[t, feat])
        )

    result = pd.Series(exposures, name="exposure")
    return result.reindex(result.abs().sort_values(ascending=False).index)
