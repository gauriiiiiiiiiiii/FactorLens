from __future__ import annotations

import pandas as pd


def long_short_factor(df: pd.DataFrame, feature: str, n_quantiles: int = 5) -> pd.Series:
    def _calc(group: pd.DataFrame) -> float:
        group = group.sort_values(feature)
        q = len(group) // n_quantiles
        if q == 0:
            return 0.0
        long_ret = group.tail(q)["return_next"].mean()
        short_ret = group.head(q)["return_next"].mean()
        return float(long_ret - short_ret)

    return df.groupby("date").apply(_calc)


def compute_factor_returns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({feature: long_short_factor(df, feature) for feature in feature_cols})


def long_short_by_prediction(
    df: pd.DataFrame,
    pred_col: str = "pred",
    ret_col: str = "return_next",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    def _calc(group: pd.DataFrame) -> float:
        group = group.sort_values(pred_col)
        q = len(group) // n_quantiles
        if q == 0:
            return 0.0
        long_ret = group.tail(q)[ret_col].mean()
        short_ret = group.head(q)[ret_col].mean()
        return float(long_ret - short_ret)

    returns = df.groupby("date").apply(_calc).rename("long_short").to_frame()
    returns["cumulative"] = (1 + returns["long_short"]).cumprod() - 1
    return returns
