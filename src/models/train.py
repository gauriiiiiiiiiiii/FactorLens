from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def train_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "return_next",
    model_choice: str = "lasso",
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_choice == "random_forest":
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=3,
            min_samples_split=6,
            max_features="sqrt",
            random_state=7,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        importance = dict(zip(feature_cols, model.feature_importances_))
    elif model_choice == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=7,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        importance = dict(zip(feature_cols, model.feature_importances_))
    else:
        model = LassoCV(cv=5, random_state=7, max_iter=10000)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        importance = dict(zip(feature_cols, model.coef_))

    report: Dict[str, float] = {
        "mse": float(mean_squared_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "test_samples": float(len(y_test)),
    }

    pred_df = df.loc[y_test.index].copy()
    pred_df["pred"] = pred

    return report, pred_df, importance
