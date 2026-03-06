from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import PROCESSED_FEATURES, PROCESSED_FACTORS
from src.data_pipeline.load_data import load_fundamentals_data, load_prices_data
from src.data_pipeline.preprocess import compute_returns, merge_price_fundamentals
from src.feature_engineering.factor_features import DEFAULT_FEATURES, available_features, build_features
from src.factor_engine.exposure_analysis import portfolio_exposure
from src.factor_engine.factor_portfolio import compute_factor_returns, long_short_by_prediction
from src.models.train import train_models
from src.visualization.dashboard import plot_factor_returns, plot_importance, plot_factor_correlation, plot_model_comparison

st.set_page_config(page_title="FactorLens", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --ink: #0e1518;
            --muted: #5c6b73;
            --accent: #e07a5f;
            --accent-2: #3d405b;
            --bg: #f6f4f0;
            --panel: #ffffff;
            --line: #e8e1d8;
        }

        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif !important;
            background: var(--bg);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            border-radius: 24px;
            padding: 2.5rem 2.75rem;
            background: radial-gradient(1200px 300px at 10% -10%, #f9e2d3 0%, transparent 70%),
                        radial-gradient(900px 300px at 90% 0%, #e0e7ff 0%, transparent 60%),
                        var(--panel);
            border: 1px solid var(--line);
            box-shadow: 0 20px 60px rgba(12, 15, 20, 0.08);
            animation: rise 600ms ease-out;
        }

        .hero h1 {
            font-size: 3rem;
            margin-bottom: 0.35rem;
            color: var(--ink);
        }

        .hero p {
            margin: 0;
            font-size: 1.15rem;
            color: var(--muted);
        }

        .pill {
            display: inline-flex;
            gap: 0.45rem;
            align-items: center;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: #fff7ef;
            color: #8a4f39;
            font-weight: 600;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 12px 30px rgba(12, 15, 20, 0.06);
            animation: float-in 700ms ease-out;
        }

        .card h3 {
            margin: 0 0 0.35rem 0;
            font-size: 1.1rem;
            color: var(--ink);
        }

        .card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .card strong {
            color: var(--ink);
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: var(--ink);
        }

        .mono {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--accent-2);
            font-size: 0.85rem;
        }

        .cta {
            background: var(--accent);
            color: white;
            padding: 0.75rem 1.2rem;
            border-radius: 12px;
            display: inline-block;
            font-weight: 600;
        }

        @keyframes rise {
            from { transform: translateY(12px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes float-in {
            from { transform: translateY(10px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="pill">FACTOR DISCOVERY PLATFORM</div>
        <h1>FactorLens</h1>
        <p>Learn return-driving factors from data, explain portfolios, and stress-test market regimes in one clean workflow.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


def _clean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def _parse_portfolio(raw: str) -> dict[str, float]:
    portfolio: dict[str, float] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) != 2:
            continue
        ticker, weight = parts
        try:
            portfolio[ticker.upper()] = float(weight)
        except ValueError:
            continue
    return portfolio


def _equal_weight_portfolio(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    weight = 1.0 / len(tickers)
    return {ticker: weight for ticker in tickers}


def _market_regime(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    if "return" in df.columns and df["return"].notna().any():
        ret_col = "return"
    else:
        ret_col = "return_next"

    daily = df.groupby("date")[ret_col].mean().rename("market_return").to_frame()
    daily["roll_mean"] = daily["market_return"].rolling(window).mean()
    daily["roll_vol"] = daily["market_return"].rolling(window).std()

    mean_med = daily["roll_mean"].median()
    vol_med = daily["roll_vol"].median()

    def _label(row: pd.Series) -> str:
        if pd.isna(row["roll_mean"]) or pd.isna(row["roll_vol"]):
            return "n/a"
        if row["roll_mean"] >= mean_med and row["roll_vol"] <= vol_med:
            return "risk-on"
        if row["roll_mean"] < mean_med and row["roll_vol"] > vol_med:
            return "risk-off"
        return "mixed"

    daily["regime"] = daily.apply(_label, axis=1)
    return daily


with st.sidebar:
    st.header("Data")
    data_source = st.selectbox("Source", ["Kaggle CSVs", "Processed CSV only"], index=1)
    st.caption("Demo processed files are included. Add Kaggle CSVs to run the full pipeline.")
    max_tickers = st.slider("Max tickers to load", min_value=50, max_value=1000, value=200, step=50)

    st.header("Model")
    model_choice = st.selectbox("Model", ["lasso", "random_forest", "xgboost"], index=0)
    st.caption("Use LASSO for sparsity, RF/XGBoost for non-linear signals.")

    st.header("Visuals")
    show_factors = st.multiselect("Factors to chart", options=DEFAULT_FEATURES, default=DEFAULT_FEATURES)

    run = st.button("Run pipeline", use_container_width=True)

st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

top_left, top_right, top_third = st.columns([1.3, 1, 1])

with top_left:
    st.markdown(
        """
        <div class="card">
            <h3>What you get</h3>
            <p>Learned factor loadings, long-short factor returns, and a quick backtest to validate signal strength.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        """
        <div class="card">
            <h3>Data mode</h3>
            <p>Use Kaggle CSVs for full pipeline, or bundled processed data for quick runs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_third:
    st.markdown(
        """
        <div class="card">
            <h3>Pipeline controls</h3>
            <p><span class="mono">{mode}</span> | features: {count}</p>
        </div>
        """.format(mode=data_source, count=len(DEFAULT_FEATURES)),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

if run:
    with st.spinner("Loading raw data..."):
        if data_source == "Kaggle CSVs":
            try:
                prices = load_prices_data(max_tickers=int(max_tickers))
                fundamentals = load_fundamentals_data()
            except (FileNotFoundError, ValueError) as exc:
                st.error(f"Kaggle CSVs missing or invalid: {exc}")
                st.info("Add CSVs under data/raw/prices and data/raw/fundamentals, then rerun.")
                st.stop()

            prices = compute_returns(prices)
            merged = merge_price_fundamentals(prices, fundamentals)
            df = build_features(merged)
            feature_cols = available_features(df)
            if not feature_cols:
                st.error("No usable features found. Check column mappings or dataset coverage.")
                st.stop()

            factor_returns = compute_factor_returns(df, feature_cols)

            PROCESSED_FEATURES.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(PROCESSED_FEATURES, index=False)
            factor_returns.to_csv(PROCESSED_FACTORS, index=True)
        else:
            if not PROCESSED_FEATURES.exists() or not PROCESSED_FACTORS.exists():
                st.error("Processed CSVs not found. Run with Kaggle CSVs once to create them.")
                st.stop()
            df = pd.read_csv(PROCESSED_FEATURES)
            factor_returns = pd.read_csv(PROCESSED_FACTORS, index_col=0)
            feature_cols = available_features(df)
            if not feature_cols:
                st.error("Processed CSV has no usable feature columns.")
                st.stop()

    with st.spinner("Training models..."):
        df_model = _clean_features(df, feature_cols)
        model_report, pred_df, importance = train_models(
            df_model,
            feature_cols=feature_cols,
            target_col="return_next",
            model_choice=model_choice,
        )
        reports = {"lasso": {}, "random_forest": {}, "xgboost": {}}
        model_preds = {}
        for name in reports:
            report, pred, _ = train_models(
                df_model,
                feature_cols=feature_cols,
                target_col="return_next",
                model_choice=name,
            )
            reports[name] = report
            model_preds[name] = pred

    with st.spinner("Backtesting..."):
        returns = long_short_by_prediction(pred_df)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Live Results</div>", unsafe_allow_html=True)

    std = returns["long_short"].std()
    sharpe = (returns["long_short"].mean() / std) * np.sqrt(252) if std and not np.isnan(std) else float("nan")
    total_return = returns["cumulative"].iloc[-1]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Test MSE", f"{model_report['mse']:.4f}")
    metric_cols[1].metric("Test R2", f"{model_report['r2']:.3f}")
    metric_cols[2].metric("Total Return", f"{total_return:.2%}")
    metric_cols[3].metric("Sharpe (ann.)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "n/a")

    tab_overview, tab_factors, tab_model, tab_backtest, tab_exposure = st.tabs(
        ["Overview", "Factors", "Model", "Backtest", "Exposure"]
    )

    with tab_overview:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Factor returns snapshot")
            st.dataframe(factor_returns.tail(12), use_container_width=True)
        with col_b:
            st.subheader("Feature sample")
            st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Cumulative factor returns")
        selected = [factor for factor in show_factors if factor in factor_returns.columns]
        factors_to_plot = factor_returns[selected] if selected else factor_returns
        st.plotly_chart(plot_factor_returns(factors_to_plot), use_container_width=True)

        st.subheader("Regime monitor")
        regime_df = _market_regime(df)
        latest_regime = regime_df["regime"].iloc[-1]
        latest_mean = regime_df["roll_mean"].iloc[-1]
        latest_vol = regime_df["roll_vol"].iloc[-1]
        r_cols = st.columns(3)
        r_cols[0].metric("Regime", latest_regime)
        r_cols[1].metric("Rolling mean", f"{latest_mean:.4f}" if np.isfinite(latest_mean) else "n/a")
        r_cols[2].metric("Rolling vol", f"{latest_vol:.4f}" if np.isfinite(latest_vol) else "n/a")
        st.line_chart(regime_df[["roll_mean", "roll_vol"]])

    with tab_factors:
        st.subheader("Factor returns by signal")
        st.dataframe(factor_returns.tail(20), use_container_width=True)
        st.plotly_chart(plot_factor_returns(factor_returns), use_container_width=True)
        st.subheader("Factor correlation")
        st.plotly_chart(plot_factor_correlation(factor_returns), use_container_width=True)

    with tab_model:
        st.subheader("Model report")
        st.json(model_report)
        st.subheader("Model comparison")
        st.plotly_chart(plot_model_comparison(reports), use_container_width=True)
        st.subheader("Backtest comparison")
        comparison = {}
        for name, pred in model_preds.items():
            returns_cmp = long_short_by_prediction(pred)
            comparison[name] = returns_cmp["cumulative"]
        comparison_df = pd.DataFrame(comparison)
        st.line_chart(comparison_df)
        st.subheader("Feature importance")
        st.dataframe(
            {
                "feature": list(importance.keys()),
                "importance": list(importance.values()),
            },
            use_container_width=True,
        )
        st.plotly_chart(plot_importance(importance), use_container_width=True)

    with tab_backtest:
        st.subheader("Long-short cumulative return")
        st.line_chart(returns["cumulative"])
        st.subheader("Return series")
        st.dataframe(returns, use_container_width=True)

    with tab_exposure:
        st.subheader("Portfolio exposure")
        tickers = sorted(df["ticker"].dropna().unique().tolist())
        defaults = [t for t in ["AAPL", "MSFT", "NVDA"] if t in tickers]
        if not defaults:
            defaults = tickers[:3]
        selected = st.multiselect("Select stocks", tickers, default=defaults)
        equal_portfolio = _equal_weight_portfolio(selected)

        with st.expander("Custom weights (optional)"):
            st.caption("Provide one ticker and weight per line, example: AAPL,0.4")
            raw_portfolio = st.text_area("Portfolio input", value="AAPL,0.4\nMSFT,0.3\nNVDA,0.3")
            parsed = _parse_portfolio(raw_portfolio)

        portfolio = parsed if parsed else equal_portfolio
        if not portfolio:
            st.info("Select at least one ticker or provide custom weights to compute exposures.")
        else:
            exposures = portfolio_exposure(df, portfolio, feature_cols)
            st.dataframe(exposures.to_frame("exposure"), use_container_width=True)

else:
    st.markdown("<div class='section-title'>Ready to run</div>", unsafe_allow_html=True)
    st.info("Set your data window and model settings, then run the pipeline to generate factors and backtest results.")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
st.caption("Demo only. Replace synthetic data with real datasets for production research.")
