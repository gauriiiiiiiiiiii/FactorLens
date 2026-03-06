from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_factor_returns(factor_returns: pd.DataFrame):
    df = factor_returns.cumsum().reset_index().rename(columns={"index": "date"})
    fig = px.line(df, x="date", y=df.columns[1:], title="Cumulative Factor Returns")
    fig.update_layout(legend_title_text="Factor")
    return fig


def plot_importance(importance: dict):
    df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
    fig = px.bar(df, x="feature", y="importance", title="Feature Importance")
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def plot_factor_correlation(factor_returns: pd.DataFrame):
    corr = factor_returns.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Factor Correlation Heatmap",
    )
    fig.update_layout(xaxis_title="Factor", yaxis_title="Factor")
    return fig


def plot_model_comparison(reports: dict[str, dict[str, float]]):
    rows = []
    for name, report in reports.items():
        rows.append({"model": name, "metric": "mse", "value": report["mse"]})
        rows.append({"model": name, "metric": "r2", "value": report["r2"]})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model Comparison (MSE, R2)",
    )
    fig.update_layout(yaxis_title="Score", xaxis_title="Model")
    return fig
