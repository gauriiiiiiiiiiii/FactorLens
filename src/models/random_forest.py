from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=7,
        n_jobs=-1,
    )
