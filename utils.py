"""Shared helpers: market data, moment estimation, transaction costs and
conversion of continuous weights into whole IDX lots (1 lot = 100 shares)."""
import numpy as np
import pandas as pd
import yfinance as yf

LOT_SIZE = 100
RISK_FREE_RATE = 0.05  # annual, roughly the Bank Indonesia policy rate


def _close(df):
    """yfinance returns either a flat or a (field, ticker) MultiIndex frame."""
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"]
    return df[["Close"]]


def load_data(assets, period):
    """Unadjusted (but split-adjusted) closing prices.

    `auto_adjust=False` matters: with the yfinance default the 'Close' column
    is back-adjusted for dividends, so historical prices are lower than the
    prices actually paid and lot counts in the backtest would be wrong.
    Dividends are therefore ignored (conservative), exactly like the ^JKSE
    price index used as benchmark.
    """
    raw = yf.download(list(assets), period=period, auto_adjust=False, progress=False)
    data = _close(raw)
    if isinstance(data, pd.Series):
        data = data.to_frame(assets[0])
    data = data.dropna(axis=1, how="all").ffill().dropna()
    return data


def load_index(ticker, period):
    raw = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    s = _close(raw)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.rename(ticker)


def estimate_moments(prices, ann_factor):
    """Annualised mean returns and Ledoit-Wolf shrunk covariance.

    With 12-36 observations and up to 30 assets the sample covariance is
    badly conditioned (singular when n_obs <= n_assets); Ledoit-Wolf
    shrinkage keeps it positive definite and much more stable out of sample.
    """
    from sklearn.covariance import LedoitWolf

    returns = prices.pct_change().dropna()
    mu = returns.mean().values * ann_factor
    if len(returns) > 1:
        cov = LedoitWolf().fit(returns.values).covariance_ * ann_factor
    else:
        cov = np.eye(prices.shape[1]) * 1e-4
    return mu, cov


def calculate_transaction_cost(k_prev, k_target, prices, buy_fee, sell_fee):
    """Asymmetric cost of moving from k_prev lots to k_target lots (eq. 3.1)."""
    k_prev = np.zeros_like(k_target) if k_prev is None else np.asarray(k_prev)
    lot_value = LOT_SIZE * np.asarray(prices)
    bought = np.maximum(k_target - k_prev, 0)
    sold = np.maximum(k_prev - k_target, 0)
    return buy_fee * np.sum(lot_value * bought) + sell_fee * np.sum(lot_value * sold)


def cash_after(k_prev, k_target, prices, cash_before, buy_fee, sell_fee):
    lot_value = LOT_SIZE * np.asarray(prices)
    trade_value = np.sum((k_target - k_prev) * lot_value)
    tc = calculate_transaction_cost(k_prev, k_target, prices, buy_fee, sell_fee)
    return cash_before - trade_value - tc, tc


def weights_to_lots(weights, wealth, prices, k_prev, buy_fee, sell_fee, max_weight=1.0):
    """Turn target weights into a feasible whole-lot portfolio.

    1. floor to whole lots,
    2. if fees make cash negative, sell the most over-weight lot until feasible,
    3. greedily buy one lot of the most under-weight asset while it fits in the
       remaining cash (fees included) and respects the max weight.

    `wealth` = cash + market value of k_prev. Returns (lots, cash_left, fees).
    The result never borrows (cash_left >= 0).
    """
    prices = np.asarray(prices, dtype=float)
    k_prev = np.asarray(k_prev, dtype=float)
    n = len(prices)
    lot_value = LOT_SIZE * prices
    cash_before = wealth - np.sum(k_prev * lot_value)
    cap = max(max_weight, 1.0 / n) * wealth

    w = np.clip(np.nan_to_num(weights), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.zeros(n)
    target_value = w * wealth
    lots = np.minimum(np.floor(target_value / lot_value), np.floor((cap + 1e-9) / lot_value))

    cash, tc = cash_after(k_prev, lots, prices, cash_before, buy_fee, sell_fee)
    while cash < 0 and lots.sum() > 0:
        over = np.where(lots > 0, lots * lot_value - target_value, -np.inf)
        lots[np.argmax(over)] -= 1
        cash, tc = cash_after(k_prev, lots, prices, cash_before, buy_fee, sell_fee)

    for _ in range(4 * n):
        deficit = target_value - lots * lot_value
        order = np.argsort(-deficit)
        bought = False
        for i in order:
            if deficit[i] < 0.5 * lot_value[i]:
                break
            if (lots[i] + 1) * lot_value[i] > cap + 1e-9:
                continue
            trial = lots.copy()
            trial[i] += 1
            c, t = cash_after(k_prev, trial, prices, cash_before, buy_fee, sell_fee)
            if c >= 0:
                lots, cash, tc, bought = trial, c, t, True
                break
        if not bought:
            break
    return lots.astype(int), cash, tc
