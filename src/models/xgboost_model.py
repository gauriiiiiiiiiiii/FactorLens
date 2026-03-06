from __future__ import annotations

import xgboost as xgb


def build_xgboost() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=7,
    )
