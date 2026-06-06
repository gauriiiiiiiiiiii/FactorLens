from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def _model(choice: str) -> Any:
    if choice == "random_forest":
        return RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                     min_samples_split=6, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    if choice == "xgboost":
        return xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.85,
                                 reg_alpha=0.1, reg_lambda=1.0,
                                 random_state=42, n_jobs=-1, verbosity=0)
    return LassoCV(cv=5, random_state=42, max_iter=10_000, n_jobs=-1)


def run(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "return_next",
    model_choice: str = "lasso",
) -> tuple[dict[str, float], pd.DataFrame, dict[str, float]]:
    X, y, idx = df[feature_cols].values, df[target_col].values, df.index
    split = int(len(X) * 0.80)

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    idx_te = idx[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    m        = _model(model_choice)
    is_lasso = isinstance(m, LassoCV)

    if is_lasso:
        m.fit(X_tr_s, y_tr);  pred = m.predict(X_te_s);  imp = dict(zip(feature_cols, m.coef_))
    else:
        m.fit(X_tr, y_tr);    pred = m.predict(X_te);    imp = dict(zip(feature_cols, m.feature_importances_))

    ic, _ = stats.spearmanr(y_te, pred)
    report = {"mse": float(mean_squared_error(y_te, pred)),
              "r2":  float(r2_score(y_te, pred)),
              "ic":  float(ic),
              "test_samples": float(len(y_te))}

    pred_df = df.loc[idx_te].copy()
    pred_df["pred"] = pred
    return report, pred_df, imp
