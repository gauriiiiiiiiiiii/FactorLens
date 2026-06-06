from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config   import PROCESSED_FEATURES, PROCESSED_FACTORS
from src.features import ALL_FEATURES, available
from src.portfolio  import factor_returns, ic_series, backtest
from src.exposure   import portfolio_exposure
from src.train      import run as train
from src.charts     import (
    cumulative_factors, importance_bar, correlation_heatmap,
    model_comparison, backtest_chart, rolling_ic, regime_chart,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="FactorLens", page_icon="📐", layout="wide")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:    #0e1518;  --muted: #5c6b73;
    --accent: #e07a5f;  --acc2:  #3d405b;
    --green:  #4caf8a;  --red:   #e05f5f;
    --bg:     #f7f5f2;  --panel: #ffffff;
    --border: #e4ddd4;  --shadow: 0 4px 20px rgba(14,21,24,.07);
}
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: var(--bg) !important;
}
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }

.hero {
    background: radial-gradient(ellipse 900px 260px at 0% -20%, #fce8df, transparent 65%),
                radial-gradient(ellipse 700px 220px at 100% 0%,  #e3e8ff, transparent 60%),
                var(--panel);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 2.2rem 2.6rem 2rem; box-shadow: var(--shadow); animation: fadeUp 500ms ease-out;
}
.hero-tag {
    display: inline-block; background: #fff4ee; color: #8a4830;
    border: 1px solid #f5d4c5; border-radius: 999px;
    padding: .28rem .85rem; font-size: .72rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; margin-bottom: .7rem;
}
.hero h1 { font-size: 2.6rem; font-weight: 700; color: var(--ink); margin: 0 0 .4rem; }
.hero p  { font-size: 1.05rem; color: var(--muted); margin: 0; max-width: 60ch; }

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: .85rem; margin: 1.2rem 0; }
.kpi {
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.1rem 1.25rem; box-shadow: var(--shadow); animation: fadeUp 600ms ease-out;
}
.kpi-label { font-size: .72rem; color: var(--muted); font-weight: 700; letter-spacing: .05em; text-transform: uppercase; margin-bottom: .3rem; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: var(--ink); line-height: 1; }
.kpi-sub   { font-size: .76rem; color: var(--muted); margin-top: .2rem; }
.kpi-good  { color: var(--green); }
.kpi-bad   { color: var(--red); }

.card { background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
        padding: 1.15rem 1.25rem; box-shadow: var(--shadow); height: 100%; }
.card-title { font-size: .9rem; font-weight: 600; color: var(--ink); margin: 0 0 .3rem; }
.card-body  { font-size: .88rem; color: var(--muted); margin: 0; line-height: 1.55; }
.mono { font-family: 'IBM Plex Mono', monospace; font-size: .82rem; color: var(--acc2); }

[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h3 {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted);
    margin: 1.1rem 0 .4rem; padding-bottom: .3rem; border-bottom: 1px solid var(--border);
}
.stButton > button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; transition: opacity .15s;
}
.stButton > button:hover { opacity: .88 !important; }

@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Factor Discovery Platform</div>
  <h1>FactorLens</h1>
  <p style="white-space: nowrap;">Learn return-predicting signals from market data, build long-short factor portfolios, and validate them with walk-forward backtests.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## FactorLens")

    st.markdown("### Model")
    model_choice = st.selectbox(
        "Algorithm",
        ["lasso", "random_forest", "xgboost"],
        format_func=lambda x: {"lasso": "LASSO", "random_forest": "Random Forest", "xgboost": "XGBoost"}[x],
    )
    n_quantiles = st.slider("Long-short quantiles", 3, 10, 5)

    st.markdown("### Visuals")
    show_factors = st.multiselect("Factors to display", ALL_FEATURES, default=ALL_FEATURES[:6])

    st.markdown("---")
    run_btn = st.button("▶  Run Pipeline", use_container_width=True)

# ── Info cards ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown("""<div class="card"><div class="card-title">15 Factors</div>
<div class="card-body">Momentum, reversal, volatility, value, quality, growth, liquidity — cross-sectionally z-scored each day.</div></div>""", unsafe_allow_html=True)
c2.markdown("""<div class="card"><div class="card-title">3 Models</div>
<div class="card-body">LASSO for sparsity, Random Forest and XGBoost for non-linear signals. Evaluated on MSE, R², Spearman IC.</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="card"><div class="card-title">Active Config</div>
<div class="card-body">Model: <span class="mono">{model_choice}</span><br>
Quantiles: <span class="mono">{n_quantiles}</span></div></div>""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    st.info("Set your options in the sidebar and click **Run Pipeline**.")
    st.stop()

# ── Helpers ──────────────────────────────────────────────────────────────────
def _clean(df, cols):
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def _parse_port(raw):
    out = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) == 2:
            try: out[parts[0].upper()] = float(parts[1])
            except ValueError: pass
    return out

def _eq_weight(tickers):
    return {t: 1/len(tickers) for t in tickers} if tickers else {}

def _regime(df, window=63):
    col   = "return" if ("return" in df.columns and df["return"].notna().any()) else "return_next"
    daily = df.groupby("date")[col].mean().rename("market_return").to_frame()
    daily["roll_mean"] = daily["market_return"].rolling(window).mean()
    daily["roll_vol"]  = daily["market_return"].rolling(window).std()
    mm, vm = daily["roll_mean"].median(), daily["roll_vol"].median()
    def _lbl(r):
        if pd.isna(r["roll_mean"]): return "n/a"
        if r["roll_mean"] >= mm and r["roll_vol"] <= vm: return "risk-on 🟢"
        if r["roll_mean"] <  mm and r["roll_vol"] >  vm: return "risk-off 🔴"
        return "mixed 🟡"
    daily["regime"] = daily.apply(_lbl, axis=1)
    return daily

def _f(v, d=3):   return f"{v:.{d}f}"  if np.isfinite(v) else "n/a"
def _p(v):        return f"{v:.2%}"    if np.isfinite(v) else "n/a"

def _kpi(label, value, sub="", cls=""):
    return f"""<div class="kpi">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value {cls}">{value}</div>
    {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
    </div>"""

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    if not PROCESSED_FEATURES.exists() or not PROCESSED_FACTORS.exists():
        st.error("Processed data not found. Place stock_features.csv and factor_returns.csv in data/processed/.")
        st.stop()
    df        = pd.read_csv(PROCESSED_FEATURES)
    fret      = pd.read_csv(PROCESSED_FACTORS, index_col=0, parse_dates=True)
    feat_cols = available(df)
    if not feat_cols:
        st.error("Processed CSV has no usable feature columns."); st.stop()

# ── Train ────────────────────────────────────────────────────────────────────
with st.spinner("Training models…"):
    df_m = _clean(df, feat_cols)
    report, pred_df, imp = train(df_m, feat_cols, model_choice=model_choice)
    all_reports, all_preds = {}, {}
    for name in ["lasso", "random_forest", "xgboost"]:
        r, p, _ = train(df_m, feat_cols, model_choice=name)
        all_reports[name] = r; all_preds[name] = p

# ── Backtest & IC ─────────────────────────────────────────────────────────────
with st.spinner("Backtesting…"):
    bt    = backtest(pred_df, n_quantiles=n_quantiles)
    ic_df = ic_series(df_m, feat_cols)

# ── KPI row ──────────────────────────────────────────────────────────────────
tot = bt["cumulative"].iloc[-1]
dd  = bt["drawdown"].min()
std = bt["long_short"].std()
sh  = (bt["long_short"].mean() / std) * np.sqrt(252) if std > 0 else float("nan")
ic_m = ic_df.mean().mean()

st.markdown(f"""<div class="kpi-grid">
{_kpi("Total Return",   _p(tot),    "Long-short",   "kpi-good" if tot>=0 else "kpi-bad")}
{_kpi("Sharpe (ann.)", _f(sh,2),   "√252 scaling")}
{_kpi("Max Drawdown",  _p(dd),     "Peak-to-trough","kpi-bad" if dd<-.1 else "")}
{_kpi("Mean IC",       _f(ic_m,4), "Spearman",      "kpi-good" if ic_m>=.02 else "")}
</div>
<div class="kpi-grid">
{_kpi("Test MSE",  _f(report["mse"],5), model_choice)}
{_kpi("Test R²",   _f(report["r2"],3),  f"{int(report['test_samples']):,} samples")}
{_kpi("IC",        _f(report["ic"],4),  "Test set")}
{_kpi("Features",  str(len(feat_cols)), "active")}
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🔬 Factors", "🤖 Model", "📈 Backtest", "💼 Portfolio"])

# Overview
with t1:
    ca, cb = st.columns(2)
    with ca:
        st.subheader("Factor Returns — last 20 days")
        st.dataframe(fret.tail(20).style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=None),
                     use_container_width=True)
    with cb:
        st.subheader("Feature sample")
        dcols = ["date","ticker"] + feat_cols[:8]
        st.dataframe(df[[c for c in dcols if c in df.columns]].head(20), use_container_width=True)

    sel = [f for f in show_factors if f in fret.columns]
    st.subheader("Cumulative factor returns")
    st.plotly_chart(cumulative_factors(fret[sel] if sel else fret), use_container_width=True)

    st.subheader("Market regime")
    rd = _regime(df)
    r1, r2, r3 = st.columns(3)
    lat = rd.iloc[-1]
    r1.metric("Regime",       lat["regime"])
    r2.metric("Rolling mean", _f(lat["roll_mean"],5) if np.isfinite(lat["roll_mean"]) else "n/a")
    r3.metric("Rolling vol",  _f(lat["roll_vol"],5)  if np.isfinite(lat["roll_vol"])  else "n/a")
    st.plotly_chart(regime_chart(rd), use_container_width=True)

# Factors
with t2:
    cl, cr = st.columns([1.1, 1])
    with cl:
        st.subheader("Factor returns table")
        st.dataframe(fret.tail(30).style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=None),
                     use_container_width=True)
    with cr:
        st.subheader("Summary stats")
        sm = fret.agg(["mean","std","min","max"]).T
        sm.columns = ["Mean","Std","Min","Max"]
        sm["Sharpe"] = sm["Mean"] / sm["Std"] * np.sqrt(252)
        st.dataframe(sm.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Correlation heatmap")
    st.plotly_chart(correlation_heatmap(fret), use_container_width=True)

    st.subheader("Rolling IC by factor")
    st.plotly_chart(rolling_ic(ic_df), use_container_width=True)

    st.subheader("Mean IC by factor")
    ic_sm = ic_df.mean().sort_values(ascending=False).reset_index()
    ic_sm.columns = ["Feature","Mean IC"]
    st.dataframe(ic_sm.style.format({"Mean IC":"{:.4f}"}).background_gradient(cmap="RdYlGn", subset=["Mean IC"]),
                 use_container_width=True)

# Model
with t3:
    mc1, mc2 = st.columns([1, 1.4])
    with mc1:
        st.subheader("Report")
        st.json(report)
        st.subheader("Feature importance")
        imp_df = (pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
                    .sort_values("importance", key=abs, ascending=False))
        st.dataframe(imp_df.style.format({"importance":"{:.5f}"}), use_container_width=True)
    with mc2:
        st.subheader("Importance chart")
        st.plotly_chart(importance_bar(imp), use_container_width=True)

    st.subheader("Model comparison")
    st.plotly_chart(model_comparison(all_reports), use_container_width=True)

    st.subheader("Backtest comparison")
    cmp = {n: backtest(p, n_quantiles=n_quantiles)["cumulative"] for n, p in all_preds.items()}
    st.line_chart(pd.DataFrame(cmp), color=["#e07a5f","#3d405b","#81b29a"])

# Backtest
with t4:
    b1, b2, b3 = st.columns(3)
    b1.metric("Total Return",  _p(tot))
    b2.metric("Sharpe (ann.)", _f(sh,2))
    b3.metric("Max Drawdown",  _p(dd))
    st.plotly_chart(backtest_chart(bt), use_container_width=True)
    with st.expander("Daily return series"):
        st.dataframe(bt.style.format("{:.5f}").background_gradient(
            cmap="RdYlGn", subset=["long_short","cumulative"]), use_container_width=True)

# Portfolio
with t5:
    st.subheader("Factor Exposure Calculator")
    tickers  = sorted(df["ticker"].dropna().unique().tolist())
    defaults = [t for t in ["AAPL","MSFT","NVDA","GOOGL"] if t in tickers][:3] or tickers[:3]
    selected = st.multiselect("Select tickers", tickers, default=defaults)
    eq_port  = _eq_weight(selected)

    pc, wc = st.columns([1.4, 1])
    with pc:
        with st.expander("Custom weights (TICKER,weight per line)"):
            raw    = st.text_area("Portfolio", value="\n".join(f"{t},{w:.2f}" for t,w in list(eq_port.items())[:5]), height=130)
            parsed = _parse_port(raw)
        portfolio = parsed or eq_port
    with wc:
        if portfolio:
            st.dataframe(pd.DataFrame.from_dict(portfolio, orient="index", columns=["Weight"])
                           .style.format("{:.2%}"), use_container_width=True)

    if portfolio:
        exp = portfolio_exposure(df, portfolio, feat_cols)
        st.dataframe(exp.to_frame("Exposure (z-score)").style.format("{:.4f}")
                       .bar(color=["#f5c3b4","#b6ddc8"], align="mid", subset=["Exposure (z-score)"]),
                     use_container_width=True)
    else:
        st.info("Select at least one ticker.")

st.caption("FactorLens · Educational use only · Data: jacksoncrow/stock-market-dataset, cnic92/200-financial-indicators")
