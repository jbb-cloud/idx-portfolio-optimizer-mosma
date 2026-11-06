"""Candidate portfolios, market-regime indicator and walk-forward backtest."""
import numpy as np
import pandas as pd
from mosma import execute_mosma
from utils import LOT_SIZE, RISK_FREE_RATE, estimate_moments, weights_to_lots

STRATEGIES = ["Min Risk", "Max Sharpe", "Max Return", "Hybrid (MOSMA)", "Naive (1/N)"]
# Rebalance roughly once a month whatever the sampling frequency.
REBALANCE_EVERY = {12: 1, 52: 4, 252: 21}


def regime_switch(index_window, contrarian=False):
    """Weight alpha given to the *defensive* (Min Risk) portfolio.

    The last index level is compared with the window mean +/- 0.5 std:
      * well above the mean (uptrend)   -> alpha = 0.1 (90 % aggressive)
      * well below the mean (downtrend) -> alpha = 0.8 (80 % defensive)
      * in between                      -> alpha = 0.5
    This is the behaviour described in the report (chapters 5 and 7.3).
    The original internship code returned the opposite values (0.8 in an
    uptrend, 0.1 in a downtrend); `contrarian=True` reproduces it.
    """
    s = index_window.squeeze() if isinstance(index_window, pd.DataFrame) else index_window
    s = pd.Series(s).dropna()
    if len(s) < 2:
        return 0.5
    band = 0.5 * float(s.std())
    last, mean = float(s.iloc[-1]), float(s.mean())
    if last > mean + band:
        alpha = 0.1
    elif last < mean - band:
        alpha = 0.8
    else:
        return 0.5
    return 0.9 - alpha if contrarian else alpha


def build_portfolio_candidates(pareto_weights, mu, cov, prices, k_prev, total_wealth,
                               tickers, buy_fee, sell_fee, weight_limit=1.0,
                               max_cash_fraction=0.25):
    """Turn MOSMA genotypes into whole-lot portfolios with their metrics."""
    prices, k_prev = np.asarray(prices, float), np.asarray(k_prev, float)
    out = {k: [] for k in ["opt_risks", "opt_returns", "sharpe_ratios", "actual_weights_opt",
                           "opt_lots", "opt_cash", "opt_fees", "hover_texts", "opt_allocations"]}
    seen = set()
    for w in pareto_weights:
        lots, cash, tc = weights_to_lots(w, total_wealth, prices, k_prev, buy_fee, sell_fee, weight_limit)
        key = tuple(lots)
        if key in seen or cash > max_cash_fraction * total_wealth or lots.sum() == 0:
            continue
        seen.add(key)
        value = lots * LOT_SIZE * prices
        w_real = value / total_wealth                      # remaining weight = cash
        net_ret = float(mu @ w_real) - tc / total_wealth
        risk = float(np.sqrt(max(w_real @ cov @ w_real, 0.0)))
        sharpe = (net_ret - RISK_FREE_RATE) / (risk + 1e-12)

        alloc = []
        for i, t in enumerate(tickers):
            if lots[i] > 0 or k_prev[i] > 0:
                diff = int(lots[i] - k_prev[i])
                order = f"Buy {diff} lots" if diff > 0 else (f"Sell {-diff} lots" if diff < 0 else "Hold")
                alloc.append((t, int(k_prev[i]), int(lots[i]), order, 100 * w_real[i]))
        alloc.sort(key=lambda x: x[4], reverse=True)

        out["opt_risks"].append(risk)
        out["opt_returns"].append(net_ret)
        out["sharpe_ratios"].append(sharpe)
        out["actual_weights_opt"].append(w_real)
        out["opt_lots"].append(lots)
        out["opt_cash"].append(cash)
        out["opt_fees"].append(tc)
        out["opt_allocations"].append(alloc)
        out["hover_texts"].append(
            f"<b>Sharpe: {sharpe:.2f}</b><br>Net return: {net_ret*100:.1f}% | Risk: {risk*100:.1f}%"
            f"<br>Cash left: {cash:,.0f} Rp")
    return out


def pick_strategies(c, alpha):
    """Target weights of the four Pareto-based strategies."""
    w_min = c["actual_weights_opt"][int(np.argmin(c["opt_risks"]))]
    w_sharpe = c["actual_weights_opt"][int(np.argmax(c["sharpe_ratios"]))]
    w_max = c["actual_weights_opt"][int(np.argmax(c["opt_returns"]))]
    norm = lambda w: w / w.sum() if w.sum() > 0 else w
    w_hyb = (1 - alpha) * norm(w_max) + alpha * norm(w_min)
    return {"Min Risk": w_min, "Max Sharpe": w_sharpe, "Max Return": w_max, "Hybrid (MOSMA)": w_hyb}


def rolling_mosma_backtest(data, data_indice, valid_tickers, initial_wealth, buy_fee, sell_fee,
                           max_weight, ann_factor, lookback_periods=None, rebalance_every=None,
                           mosma_kwargs=None, seed=0, contrarian=False, progress=None):
    """Walk-forward backtest, starting from cash.

    At each rebalancing date t only prices up to t are used (estimation
    window = `lookback_periods`); orders are executed at the close of t and
    the portfolio is valued at the following closes. Lots, cash and fees are
    tracked separately for every strategy, and every strategy rebalances from
    its *own* holdings.
    """
    mosma_kwargs = dict(mosma_kwargs or {})
    mosma_kwargs.setdefault("pop_size", 60)
    mosma_kwargs.setdefault("max_iterations", 60)
    lookback = lookback_periods or 2 * ann_factor
    reb = rebalance_every or REBALANCE_EVERY.get(ann_factor, 1)

    prices_all = data.values.astype(float)
    idx = data_indice.reindex(data.index).ffill()
    n_assets, dates = len(valid_tickers), data.index
    if len(dates) <= lookback + 1:
        raise ValueError(f"Not enough history: {len(dates)} periods for a {lookback}-period lookback.")

    state = {s: {"lots": np.zeros(n_assets), "cash": float(initial_wealth), "fees": 0.0,
                 "turnover": 0.0} for s in STRATEGIES}
    vals = {s: [float(initial_wealth)] for s in STRATEGIES}
    alphas = []
    dates_oos = [dates[lookback - 1]]
    steps = list(range(lookback, len(dates)))

    for k, i in enumerate(steps):
        t = i - 1                                   # decision date (close known)
        p_t, p_next = prices_all[t], prices_all[i]
        if (i - lookback) % reb == 0:
            if progress:
                progress(k / len(steps), f"Rebalancing {dates[t]:%Y-%m-%d} ({k+1}/{len(steps)})")
            window = data.iloc[t - lookback + 1: t + 1]
            mu, cov = estimate_moments(window, ann_factor)
            alpha = regime_switch(idx.iloc[t - lookback + 1: t + 1], contrarian=contrarian)
            alphas.append((dates[t], alpha))

            ref_wealth = float(np.mean([st["cash"] + st["lots"] @ (LOT_SIZE * p_t) for st in state.values()]))
            genotypes = execute_mosma(n_assets, mu, cov, max_weight, p_t, np.zeros(n_assets), ref_wealth,
                                      buy_fee, sell_fee, seed=seed + k, **mosma_kwargs)
            cands = build_portfolio_candidates(genotypes, mu, cov, p_t, np.zeros(n_assets), ref_wealth,
                                               valid_tickers, buy_fee, sell_fee, max_weight)
            targets = pick_strategies(cands, alpha) if cands["sharpe_ratios"] else {}
            targets["Naive (1/N)"] = np.ones(n_assets) / n_assets

            for s, w in targets.items():
                st = state[s]
                wealth = st["cash"] + st["lots"] @ (LOT_SIZE * p_t)
                lots, cash, tc = weights_to_lots(w, wealth, p_t, st["lots"], buy_fee, sell_fee, max_weight)
                st["turnover"] += np.abs(lots - st["lots"]) @ (LOT_SIZE * p_t) / max(wealth, 1e-9)
                st["lots"], st["cash"], st["fees"] = lots, cash, st["fees"] + tc

        for s in STRATEGIES:
            st = state[s]
            vals[s].append(st["cash"] + st["lots"] @ (LOT_SIZE * p_next))
        dates_oos.append(dates[i])

    if progress:
        progress(1.0, "Backtest done")
    extra = {"fees": {s: state[s]["fees"] for s in STRATEGIES},
             "turnover": {s: state[s]["turnover"] for s in STRATEGIES},
             "alphas": alphas}
    return dates_oos, vals, extra


def performance_table(vals, ann_factor, initial_wealth, fees=None):
    rows = []
    for s, v in vals.items():
        arr = np.asarray(v, float)
        r = arr[1:] / arr[:-1] - 1
        years = len(r) / ann_factor
        cagr = (arr[-1] / arr[0]) ** (1 / years) - 1 if years > 0 and arr[-1] > 0 else np.nan
        vol = r.std(ddof=1) * np.sqrt(ann_factor) if len(r) > 1 else np.nan
        sharpe = (r.mean() * ann_factor - RISK_FREE_RATE) / vol if vol and vol > 0 else np.nan
        dd = (arr - np.maximum.accumulate(arr)) / np.maximum.accumulate(arr)
        rows.append({
            "Strategy": s,
            "Final Value (Rp)": f"{arr[-1]:,.0f}",
            "Total Return": f"{(arr[-1] / initial_wealth - 1) * 100:.1f}%",
            "CAGR": f"{cagr * 100:.1f}%",
            "Ann. Volatility": f"{vol * 100:.1f}%",
            "Sharpe (rf=5%)": f"{sharpe:.2f}",
            "Max Drawdown": f"{dd.min() * 100:.1f}%",
            "VaR 95% (per period)": f"{np.percentile(r, 5) * 100:.2f}%" if len(r) else "n/a",
            "Fees Paid (Rp)": f"{fees[s]:,.0f}" if fees and s in fees else "-",
        })
    return pd.DataFrame(rows)
