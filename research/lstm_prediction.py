"""LSTM experiment (report, chapter 8): predicting ADRO.JK's daily return.

Fixes compared with the internship version:
* the scaler is fitted on the training period only (no test-set leakage),
* no backward fill (it copied future prices into the past),
* a validation split + early stopping instead of a fixed 20 epochs,
* fixed random seeds,
* every result is compared with the naive random-walk baseline
  ("tomorrow's price = today's price", i.e. predicted return = 0).
  A MAPE is meaningless without this baseline: on daily prices the naive
  model already reaches a MAPE close to the average absolute daily return.

Run:  python research/lstm_prediction.py          (add --save to write docs/figures/result_lstm_vs_naive.png)
"""
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

SEED = 42
TARGET = "ADRO.JK"
WINDOW = 30          # days of history fed to the network
TRAIN_FRAC = 0.8
EVAL_DAYS = 60       # one-step-ahead evaluation horizon
RECURSIVE_DAYS = 7   # recursive (autoregressive) horizon

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Without this, multithreaded CPU kernels still make two runs differ slightly
# (the 1-day MAPE moved between ~2.4 % and ~3.0 % across runs; the naive
# baseline, 1.8 %, does not change).
tf.config.experimental.enable_op_determinism()

ASSETS = ["ADRO.JK", "AKRA.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK",
          "BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK", "BRPT.JK", "BUMI.JK",
          "CPIN.JK", "EMTK.JK", "ESSA.JK", "GOTO.JK", "HRUM.JK", "ICBP.JK",
          "INDF.JK", "INKP.JK", "ISAT.JK", "KLBF.JK", "MDKA.JK", "MEDC.JK",
          "PGAS.JK", "PTBA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK"]


# ---------------------------------------------------------------------------
# Data and features (multivariate: 30 stock prices + indicators of the target)
# ---------------------------------------------------------------------------
def build_features():
    raw = yf.download(ASSETS, period="5y", auto_adjust=True, progress=False)["Close"]
    data = raw.dropna(axis=1, how="all").ffill()
    # keep only stocks that already traded at the start (no backward fill)
    data = data.loc[:, data.iloc[0].notna()].dropna()

    p = data[TARGET]
    data["EMA_7"] = p.ewm(span=7, adjust=False).mean()
    data["EMA_21"] = p.ewm(span=21, adjust=False).mean()
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    data["RSI"] = 100 - 100 / (1 + gain / loss)
    sma, std = p.rolling(20).mean(), p.rolling(20).std()
    data["BB_WIDTH"] = 4 * std / sma
    data["Return"] = p.pct_change()
    return data.dropna()


def make_sequences(values, target_col, window):
    X = np.stack([values[i:i + window] for i in range(len(values) - window)])
    y = values[window:, target_col]
    return X, y


def build_model(n_features):
    model = Sequential([
        tf.keras.Input(shape=(WINDOW, n_features)),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50), Dropout(0.2),
        Dense(25), Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_absolute_error")
    return model


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main():
    data = build_features()
    cols = list(data.columns)
    t_col = cols.index("Return")

    split = int((len(data) - WINDOW) * TRAIN_FRAC) + WINDOW   # first row whose target is test
    scaler = MinMaxScaler().fit(data.iloc[:split])             # fitted on training rows only
    scaled = scaler.transform(data)
    ret_min, ret_scale = scaler.data_min_[t_col], scaler.data_range_[t_col]
    unscale = lambda z: z * ret_scale + ret_min

    X, y = make_sequences(scaled, t_col, WINDOW)
    n_train = split - WINDOW
    X_tr, y_tr, X_te, y_te = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    model = build_model(X.shape[2])
    model.fit(X_tr, y_tr, batch_size=32, epochs=200, validation_split=0.15, shuffle=False, verbose=0,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    # ---------------- test set: return prediction -------------------------
    pred_ret = unscale(model.predict(X_te, verbose=0).ravel())
    true_ret = unscale(y_te)
    mae_model = np.mean(np.abs(true_ret - pred_ret))
    mae_naive = np.mean(np.abs(true_ret))                       # predicted return = 0
    hit = np.mean(np.sign(pred_ret) == np.sign(true_ret)) * 100
    print(f"Test set ({len(true_ret)} days) – MAE of daily return: LSTM {mae_model:.4%} | "
          f"naive (0) {mae_naive:.4%} | directional accuracy {hit:.1f}%")

    # ---------------- 1-day-ahead price on the last EVAL_DAYS -------------
    prices = data[TARGET].values
    prev = prices[-EVAL_DAYS - 1:-1]
    real = prices[-EVAL_DAYS:]
    pred_last = pred_ret[-EVAL_DAYS:]
    lstm_price = prev * (1 + pred_last)
    m_lstm, m_naive = mape(real, lstm_price), mape(real, prev)
    print(f"1-day-ahead price, last {EVAL_DAYS} days – MAPE LSTM {m_lstm:.2f}% | naive {m_naive:.2f}%")

    # ---------------- recursive RECURSIVE_DAYS forecast -------------------
    # Only the target's return is fed back; the other features are frozen at
    # their last known value, which is a strong simplification.
    window = scaled[-RECURSIVE_DAYS - WINDOW:-RECURSIVE_DAYS].copy()
    price = prices[-RECURSIVE_DAYS - 1]
    rec = []
    for _ in range(RECURSIVE_DAYS):
        z = model.predict(window[None], verbose=0)[0, 0]
        price *= 1 + unscale(z)
        rec.append(price)
        new_row = window[-1].copy()
        new_row[t_col] = z
        window = np.vstack([window[1:], new_row])
    real_rec = prices[-RECURSIVE_DAYS:]
    naive_rec = np.full(RECURSIVE_DAYS, prices[-RECURSIVE_DAYS - 1])
    print(f"Recursive {RECURSIVE_DAYS}-day path – MAPE LSTM {mape(real_rec, np.array(rec)):.2f}% | "
          f"naive (flat) {mape(real_rec, naive_rec):.2f}%  (only {RECURSIVE_DAYS} points: not significant)")

    m_rec_lstm, m_rec_naive = mape(real_rec, np.array(rec)), mape(real_rec, naive_rec)
    dates = data.index
    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    d = dates[-EVAL_DAYS:]
    a1.plot(d, real, color=BLUE, lw=2, label="Prix réel")
    a1.plot(d, lstm_price, color=ORANGE, lw=1.8, label=f"LSTM (MAPE {m_lstm:.2f} %)")
    a1.plot(d, prev, color=AQUA, lw=1.8, ls="--", label=f"Naïf « demain = aujourd'hui » (MAPE {m_naive:.2f} %)")
    a1.set_title(f"{TARGET} : prévision à 1 jour, {EVAL_DAYS} derniers jours", color=INK, fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK)
    a1.tick_params(axis="x", rotation=30)
    labels = ["1 jour\n(60 j)", f"Récursif\n{RECURSIVE_DAYS} j"]
    x = np.arange(2)
    bl = a2.bar(x - 0.19, [m_lstm, m_rec_lstm], 0.36, color=ORANGE, label="LSTM")
    bn = a2.bar(x + 0.19, [m_naive, m_rec_naive], 0.36, color=AQUA, label="Naïf")
    for bars in (bl, bn):
        for b in bars:
            a2.annotate(f"{b.get_height():.1f} %", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    a2.set_xticks(x, labels)
    a2.set_ylabel("MAPE (%) – plus bas = mieux", color=MUTED, fontsize=9)
    a2.set_title("Erreur de prévision", color=INK, fontsize=11, loc="left")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    if "--save" in sys.argv:
        out = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "result_lstm_vs_naive.png")
        fig.savefig(out, dpi=160, facecolor=SURFACE)
        print("saved", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
