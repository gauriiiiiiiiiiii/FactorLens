from __future__ import annotations

from sklearn.linear_model import LassoCV


def build_lasso() -> LassoCV:
    return LassoCV(cv=5, random_state=7, max_iter=5000)
