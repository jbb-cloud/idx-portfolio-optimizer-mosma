"""Streamlit front-end: whole-lot portfolio optimisation on the IDX with MOSMA."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import (build_portfolio_candidates, performance_table, regime_switch,
                      rolling_mosma_backtest, REBALANCE_EVERY)
from mosma import execute_mosma
from utils import LOT_SIZE, estimate_moments, load_data, load_index, weights_to_lots

load_data_cached = st.cache_data(show_spinner=False, ttl=3600)(load_data)
load_index_cached = st.cache_data(show_spinner=False, ttl=3600)(load_index)

st.set_page_config(page_title="IDX Optimizer", layout="wide")
st.title("Portfolio Optimizer with Real Fees (MOSMA)")
st.markdown("Whole-lot (100 shares) portfolio optimisation for the Indonesia Stock Exchange, "
            "with asymmetric brokerage fees and your current holdings taken into account.")
st.caption("Research prototype – not investment advice. Backtests use today's IDX30 members "
           "and therefore suffer from survivorship bias.")

if "portfolio_lots" not in st.session_state:
    st.session_state.portfolio_lots = {}
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.header("Investment Parameters")
TIMEFRAME = sb.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=2)
TOTAL_BUDGET = sb.number_input("Available Cash (Rp)", min_value=0, value=10_000_000, step=1_000_000)
MAX_WEIGHT = sb.slider("Maximum weight per stock (%)", 5, 100, 100, 5) / 100.0

sb.subheader("Analysis Period")
HISTORY = sb.selectbox("Historical data", ["2y", "5y", "10y"], index=1)
LOOKBACK_YEARS = sb.selectbox("Estimation window (years)", [1, 2, 3], index=1,
                              help="Window used to estimate returns/covariance. In the backtest it "
                                   "rolls forward; in the current optimisation it is the most recent window.")

sb.subheader("Brokerage Fees")
BUY_FEE = sb.number_input("Buy Fee (%)", 0.0, 5.0, 0.15, 0.01) / 100.0
SELL_FEE = sb.number_input("Sell Fee (%)", 0.0, 5.0, 0.25, 0.01) / 100.0

sb.subheader("Algorithm Parameters")
POP_SIZE = sb.number_input("Population size", 10, 1000, 150, 10)
MAX_ITER = sb.number_input("Max iterations", 10, 2000, 150, 10)
STOP_CRITERION = sb.selectbox("Stopping criterion", ["Max Iterations", "Hypervolume Plateau"])
PATIENCE = sb.number_input("Plateau patience", 1, 100, 15) if STOP_CRITERION == "Hypervolume Plateau" else 15
USE_REPAIR = sb.checkbox("Use repair function", value=True,
                         help="Projects every candidate onto the simplex with the max-weight cap.")
SEED = sb.number_input("Random seed", 0, 10_000, 42, help="Fixes MOSMA's randomness for reproducible results.")

sb.subheader("Backtest")
RUN_BACKTEST = sb.checkbox("Run walk-forward backtest", value=True)
BT_POP = sb.number_input("Backtest population", 10, 500, 60, 10, help="Smaller than above to keep it fast.")
BT_ITER = sb.number_input("Backtest iterations", 10, 500, 60, 10)

default_assets = [
    "ADRO.JK", "AKRA.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK",
    "BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK", "BRPT.JK", "BUMI.JK",
    "CPIN.JK", "EMTK.JK", "ESSA.JK", "GOTO.JK", "HRUM.JK", "ICBP.JK",
    "INDF.JK", "INKP.JK", "ISAT.JK", "KLBF.JK", "MDKA.JK", "MEDC.JK",
    "PGAS.JK", "PTBA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
]
selected = sb.multiselect("Select stocks (min. 3)", default_assets, default=default_assets[:8])
manual = sb.text_input("Add tickers (comma-separated, Yahoo symbols, e.g. TLKM.JK)")
for t in [x.strip().upper() for x in manual.split(",") if x.strip()]:
    if t not in selected:
        selected.append(t)

sb.subheader("Current Portfolio (optional)")
st.session_state.portfolio_lots = {a: st.session_state.portfolio_lots.get(a, 0) for a in selected}
df_current = sb.data_editor(
    pd.DataFrame({"Stock": selected, "Current Lots": [st.session_state.portfolio_lots[a] for a in selected]}),
    hide_index=True, width="stretch",
    column_config={"Stock": st.column_config.TextColumn(disabled=True),
                   "Current Lots": st.column_config.NumberColumn(min_value=0, step=1)})
st.session_state.portfolio_lots = dict(zip(df_current["Stock"], df_current["Current Lots"].fillna(0).astype(int)))

run = sb.button("Run Optimization & Backtest", width="stretch", type="primary")


def progress_widget():
    bar, txt = st.progress(0.0), st.empty()

    def cb(frac, text):
        bar.progress(min(max(frac, 0.0), 1.0))
        txt.text(text)
    return cb, (bar, txt)


# ---------------------------------------------------------------------------
# Computation (only when the button is pressed; results kept in session_state
# so that clicking on the chart does not re-run everything)
# ---------------------------------------------------------------------------
if run:
    if len(selected) < 3:
        st.error("Please select at least 3 stocks.")
        st.stop()
    with st.spinner("Downloading market data..."):
        data = load_data_cached(tuple(selected), HISTORY)
        index_raw = load_index_cached("^JKSE", HISTORY)
    rule = {"Weekly": "W-FRI", "Monthly": "ME"}.get(TIMEFRAME)
    ann_factor = {"Daily": 252, "Weekly": 52, "Monthly": 12}[TIMEFRAME]
    if rule:
        data = data.resample(rule).last().dropna(how="all")
        index_raw = index_raw.resample(rule).last()
    data_index = index_raw.reindex(data.index).ffill()

    missing = sorted(set(selected) - set(data.columns))
    if missing:
        st.warning(f"No data for: {', '.join(missing)} (ignored).")
    tickers = list(data.columns)
    n = len(tickers)
    lookback = LOOKBACK_YEARS * ann_factor
    if len(data) <= lookback:
        st.error(f"Only {len(data)} periods of data: choose a longer history or a shorter window.")
        st.stop()

    prices = data.iloc[-1].values.astype(float)
    mu, cov = estimate_moments(data.iloc[-lookback:], ann_factor)
    k_prev = np.array([st.session_state.portfolio_lots.get(t, 0) for t in tickers], dtype=float)
    total_wealth = TOTAL_BUDGET + float(k_prev @ (LOT_SIZE * prices))
    if total_wealth <= 0:
        st.error("Provide either available cash or an existing portfolio.")
        st.stop()

    cb, widgets = progress_widget()
    genotypes = execute_mosma(n, mu, cov, MAX_WEIGHT, prices, k_prev, total_wealth, BUY_FEE, SELL_FEE,
                              pop_size=POP_SIZE, max_iterations=MAX_ITER, stop_criterion=STOP_CRITERION,
                              patience=PATIENCE, use_repair=USE_REPAIR, seed=SEED, progress=cb)
    for w in widgets:
        w.empty()
    cands = build_portfolio_candidates(genotypes, mu, cov, prices, k_prev, total_wealth, tickers,
                                       BUY_FEE, SELL_FEE, MAX_WEIGHT)
    alpha = regime_switch(data_index.iloc[-lookback:])

    bt = None
    if RUN_BACKTEST:
        cb, widgets = progress_widget()
        try:
            bt = rolling_mosma_backtest(
                data, data_index, tickers, total_wealth, BUY_FEE, SELL_FEE, MAX_WEIGHT, ann_factor,
                lookback_periods=lookback,
                mosma_kwargs=dict(pop_size=BT_POP, max_iterations=BT_ITER, stop_criterion=STOP_CRITERION,
                                  patience=PATIENCE, use_repair=USE_REPAIR),
                seed=SEED, progress=cb)
        except ValueError as e:
            st.warning(f"Backtest skipped: {e}")
        for w in widgets:
            w.empty()

    st.session_state.results = dict(cands=cands, alpha=alpha, prices=prices, k_prev=k_prev, mu=mu, cov=cov,
                                    total_wealth=total_wealth, tickers=tickers, data_index=data_index,
                                    ann_factor=ann_factor, bt=bt, timeframe=TIMEFRAME,
                                    fees=(BUY_FEE, SELL_FEE), max_weight=MAX_WEIGHT)

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
R = st.session_state.results
if R is None:
    st.info("Set the parameters and click **Run Optimization & Backtest**.")
    st.stop()

c = R["cands"]
if not c["sharpe_ratios"]:
    st.error("No whole-lot portfolio satisfies the constraints (budget too small for one lot, "
             "or max weight too tight). Increase the cash or relax the max weight.")
    st.stop()

prices, k_prev, tickers, total_wealth = R["prices"], R["k_prev"], R["tickers"], R["total_wealth"]
buy_fee, sell_fee = R["fees"]
alpha = R["alpha"]
risks, rets, sharpes = c["opt_risks"], c["opt_returns"], c["sharpe_ratios"]

# Hybrid portfolio, converted to lots with fees and current holdings
w_min = c["actual_weights_opt"][int(np.argmin(risks))]
w_max = c["actual_weights_opt"][int(np.argmax(rets))]
w_hyb = (1 - alpha) * w_max / w_max.sum() + alpha * w_min / w_min.sum()
lots_h, cash_h, tc_h = weights_to_lots(w_hyb, total_wealth, prices, k_prev, buy_fee, sell_fee, R["max_weight"])
wr_h = lots_h * LOT_SIZE * prices / total_wealth
ret_h = float(R["mu"] @ wr_h) - tc_h / total_wealth
risk_h = float(np.sqrt(wr_h @ R["cov"] @ wr_h))

col1, col2 = st.columns([2, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risks, y=rets, mode="markers", name="Pareto front",
                             marker=dict(size=12, color=sharpes, colorscale="Viridis", showscale=True,
                                         colorbar=dict(title="Sharpe")),
                             text=c["hover_texts"], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[risk_h], y=[ret_h], mode="markers", name="Hybrid",
                             marker=dict(size=18, color="#9b5de5", symbol="star", line=dict(width=2, color="white")),
                             text=[f"<b>Hybrid</b><br>Net return {ret_h*100:.1f}% | Risk {risk_h*100:.1f}%"],
                             hoverinfo="text"))
    fig.update_layout(title="Pareto front of whole-lot portfolios (MOSMA)", xaxis_title="Risk (ann. volatility)",
                      yaxis_title="Expected net return (ann.)", height=450, showlegend=False)
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points")

    pts = [p for p in (ev.selection.points if ev and ev.selection else []) if p.get("curve_number", 0) == 0]
    if pts:
        idx_t, label = pts[0]["point_index"], "Selected portfolio"
    else:
        idx_t, label = int(np.argmax(sharpes)), "Best Sharpe portfolio (default)"

with col2:
    st.subheader(label)
    st.metric("Sharpe ratio", f"{sharpes[idx_t]:.2f}")
    st.metric("Expected net return", f"{rets[idx_t]*100:.1f} %")
    st.metric("Risk", f"{risks[idx_t]*100:.1f} %")
    st.caption(f"Fees: {c['opt_fees'][idx_t]:,.0f} Rp · Cash left: {c['opt_cash'][idx_t]:,.0f} Rp")
    df_t = pd.DataFrame(c["opt_allocations"][idx_t], columns=["Stock", "Current Lots", "Target Lots", "Order", "Weight (%)"])
    st.dataframe(df_t.round({"Weight (%)": 1}), width="stretch", hide_index=True)
st.caption("Expected returns are historical means over the estimation window: they are noisy "
           "estimates, not forecasts.")

st.markdown("---")
st.subheader("Hybrid portfolio (regime-dependent)")
st.info(f"**Market regime alpha = {alpha:.2f}** (0 = fully aggressive Max Return, 1 = fully defensive "
        f"Min Risk). ^JKSE vs its mean over the estimation window: "
        f"{'uptrend' if alpha < 0.5 else 'downtrend' if alpha > 0.5 else 'neutral'}.")
rows = []
for i, t in enumerate(tickers):
    if lots_h[i] > 0 or k_prev[i] > 0:
        d = int(lots_h[i] - k_prev[i])
        rows.append((t, int(k_prev[i]), int(lots_h[i]),
                     f"Buy {d} lots" if d > 0 else (f"Sell {-d} lots" if d < 0 else "Hold"), 100 * wr_h[i]))
df_h = pd.DataFrame(sorted(rows, key=lambda x: -x[4]), columns=["Stock", "Current Lots", "Target Lots", "Order", "Weight (%)"])
st.dataframe(df_h.round({"Weight (%)": 1}), width="stretch", hide_index=True)
st.caption(f"Fees: {tc_h:,.0f} Rp · Cash left: {cash_h:,.0f} Rp")

if R["bt"] is not None:
    dates_oos, vals, extra = R["bt"]
    st.markdown("---")
    st.subheader(f"Walk-forward backtest ({R['timeframe']}, monthly rebalancing, starting from cash)")
    idx_vals = R["data_index"].reindex(pd.to_datetime(dates_oos)).ffill()
    vals = dict(vals)
    vals["^JKSE (price index)"] = list((idx_vals / idx_vals.iloc[0] * total_wealth).values)
    colors = {"Min Risk": "#2a9d8f", "Max Sharpe": "#e9c46a", "Max Return": "#e76f51",
              "Hybrid (MOSMA)": "#9b5de5", "Naive (1/N)": "#8d99ae", "^JKSE (price index)": "#888888"}
    fig_bt = go.Figure()
    for s, v in vals.items():
        dash = "dot" if s in ("Naive (1/N)", "^JKSE (price index)") else None
        fig_bt.add_trace(go.Scatter(x=dates_oos, y=v, mode="lines", name=s,
                                    line=dict(color=colors[s], width=4 if s.startswith("Hybrid") else 2, dash=dash)))
    fig_bt.update_layout(xaxis_title="Date", yaxis_title="Portfolio value (Rp)", hovermode="x unified", height=550,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_bt, width="stretch")
    st.markdown("#### Performance & risk")
    st.dataframe(performance_table(vals, R["ann_factor"], total_wealth, extra["fees"]),
                 width="stretch", hide_index=True)
    st.caption("Single historical path, dividends ignored, survivorship bias (today's IDX30 members). "
               "Differences between strategies are not tested for statistical significance.")
