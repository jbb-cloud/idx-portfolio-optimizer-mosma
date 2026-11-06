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

Run:  python research/lstm_prediction.py
"""
import random

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

    dates = data.index
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8))
    a1.plot(dates[-EVAL_DAYS:], real, label="Real", color="blue", marker=".")
    a1.plot(dates[-EVAL_DAYS:], lstm_price, label=f"LSTM (MAPE {m_lstm:.2f}%)", color="red", marker=".")
    a1.plot(dates[-EVAL_DAYS:], prev, label=f"Naive (MAPE {m_naive:.2f}%)", color="grey", ls="--")
    a1.set_title(f"{TARGET}: 1-day-ahead prediction")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(dates[-RECURSIVE_DAYS:], real_rec, label="Real", color="blue", marker=".")
    a2.plot(dates[-RECURSIVE_DAYS:], rec, label="LSTM recursive", color="red", marker=".")
    a2.plot(dates[-RECURSIVE_DAYS:], naive_rec, label="Naive (flat)", color="grey", ls="--")
    a2.set_title(f"{TARGET}: recursive {RECURSIVE_DAYS}-day prediction")
    a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
