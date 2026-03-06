from __future__ import annotations

from typing import Dict

import pandas as pd


def portfolio_exposure(df: pd.DataFrame, portfolio: Dict[str, float], feature_cols: list[str]) -> pd.Series:
    latest = df.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    exposures: Dict[str, float] = {}

    for feature in feature_cols:
        exposure = 0.0
        for ticker, weight in portfolio.items():
            if ticker in latest.index:
                exposure += weight * float(latest.loc[ticker, feature])
        exposures[feature] = exposure

    return pd.Series(exposures).sort_values(ascending=False)
