from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_LAYOUT = dict(
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    font=dict(family="Space Grotesk, sans-serif", color="#0e1518", size=13),
    margin=dict(t=52, b=44, l=60, r=24),
    xaxis=dict(gridcolor="#ece8e1", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#ece8e1", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ece8e1", borderwidth=1),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#ece8e1", font_size=13),
)
_PAL = px.colors.qualitative.Set2


def _apply(fig: go.Figure) -> go.Figure:
    return fig.update_layout(**_LAYOUT)


def cumulative_factors(factor_df: pd.DataFrame) -> go.Figure:
    cum = factor_df.cumsum().reset_index()
    date_col = cum.columns[0]
    fig = go.Figure()
    for i, col in enumerate(cum.columns[1:]):
        fig.add_trace(go.Scatter(x=cum[date_col], y=cum[col], name=col,
                                  mode="lines", line=dict(width=1.8, color=_PAL[i % len(_PAL)]),
                                  hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>"))
    fig.update_layout(title="Cumulative Factor Returns", xaxis_title="Date",
                      yaxis_title="Cumulative Return", hovermode="x unified")
    return _apply(fig)


def importance_bar(importance: dict[str, float]) -> go.Figure:
    df = (pd.DataFrame({"feature": list(importance.keys()), "v": list(importance.values())})
            .assign(abs_v=lambda d: d["v"].abs()).sort_values("abs_v"))
    colors = ["#e07a5f" if v >= 0 else "#3d405b" for v in df["v"]]
    fig = go.Figure(go.Bar(x=df["v"], y=df["feature"], orientation="h",
                            marker_color=colors,
                            hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"))
    fig.update_layout(title="Feature Importance", xaxis_title="Score", showlegend=False)
    return _apply(fig)


def correlation_heatmap(factor_df: pd.DataFrame) -> go.Figure:
    corr = factor_df.corr().round(2)
    fig  = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig.update_layout(title="Factor Correlation Heatmap")
    return _apply(fig)


def model_comparison(reports: dict[str, dict[str, float]]) -> go.Figure:
    metrics = [("mse", "#e07a5f"), ("r2", "#3d405b"), ("ic", "#81b29a")]
    fig = go.Figure()
    for metric, color in metrics:
        fig.add_trace(go.Bar(name=metric.upper(),
                              x=list(reports.keys()),
                              y=[reports[m].get(metric, 0) for m in reports],
                              marker_color=color,
                              hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"))
    fig.update_layout(barmode="group", title="Model Comparison", xaxis_title="Model", yaxis_title="Score")
    return _apply(fig)


def backtest_chart(returns: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.06, subplot_titles=("Cumulative Return", "Drawdown"))
    fig.add_trace(go.Scatter(x=returns.index, y=returns["cumulative"], name="Long-Short",
                              mode="lines", line=dict(width=2, color="#3d405b"),
                              fill="tozeroy", fillcolor="rgba(61,64,91,0.08)",
                              hovertemplate="Date: %{x}<br>Return: %{y:.2%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=returns.index, y=returns["drawdown"], name="Drawdown",
                              mode="lines", line=dict(width=1.5, color="#e07a5f"),
                              fill="tozeroy", fillcolor="rgba(224,122,95,0.15)",
                              hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    base = {k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(title="Long-Short Backtest", showlegend=False, hovermode="x unified", **base)
    return fig


def rolling_ic(ic_df: pd.DataFrame, window: int = 21) -> go.Figure:
    roll = ic_df.rolling(window).mean()
    mean = roll.mean(axis=1)
    fig  = go.Figure()
    for col in ic_df.columns:
        fig.add_trace(go.Scatter(x=roll.index, y=roll[col], name=col,
                                  mode="lines", line=dict(width=1.2), opacity=0.6))
    fig.add_trace(go.Scatter(x=mean.index, y=mean, name="Avg IC",
                              mode="lines", line=dict(width=2.5, color="#0e1518", dash="dash")))
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig.update_layout(title=f"Rolling {window}-Day IC", xaxis_title="Date",
                      yaxis_title="IC (Spearman)", hovermode="x unified")
    return _apply(fig)


def regime_chart(regime_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=regime_df.index, y=regime_df["roll_mean"],
                              name="Rolling Mean Return", line=dict(color="#3d405b", width=1.8)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=regime_df.index, y=regime_df["roll_vol"],
                              name="Rolling Volatility",  line=dict(color="#e07a5f", width=1.8, dash="dot")),
                  secondary_y=True)
    fig.update_yaxes(title_text="Mean Return", secondary_y=False)
    fig.update_yaxes(title_text="Volatility",  secondary_y=True)
    base = {k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(title="Market Regime Monitor", hovermode="x unified", **base)
    return fig
