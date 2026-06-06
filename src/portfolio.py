from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def factor_returns(df: pd.DataFrame, feature_cols: list[str], n_quantiles: int = 5) -> pd.DataFrame:
    def _ls(group: pd.DataFrame, col: str) -> float:
        g = group.dropna(subset=[col, "return_next"])
        n = len(g) // n_quantiles
        if n == 0:
            return 0.0
        s = g.sort_values(col)
        return float(s.tail(n)["return_next"].mean() - s.head(n)["return_next"].mean())

    return pd.DataFrame(
        {f: df.groupby("date").apply(_ls, col=f) for f in feature_cols}
    )


def ic_series(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for date, grp in df.groupby("date"):
        row: dict = {"date": date}
        for col in feature_cols:
            sub = grp[[col, "return_next"]].dropna()
            row[col] = float(stats.spearmanr(sub[col], sub["return_next"]).correlation) if len(sub) >= 5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def backtest(
    df: pd.DataFrame,
    pred_col: str = "pred",
    ret_col: str = "return_next",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    daily = []
    for date, grp in df.groupby("date"):
        g = grp.dropna(subset=[pred_col, ret_col])
        n = len(g) // n_quantiles
        if n == 0:
            daily.append((date, 0.0))
            continue
        s = g.sort_values(pred_col)
        daily.append((date, float(s.tail(n)[ret_col].mean() - s.head(n)[ret_col].mean())))

    result = pd.DataFrame(daily, columns=["date", "long_short"]).set_index("date")
    result["cumulative"] = (1 + result["long_short"]).cumprod() - 1
    wealth = (1 + result["long_short"]).cumprod()
    result["drawdown"]   = (wealth - wealth.cummax()) / wealth.cummax()
    return result
