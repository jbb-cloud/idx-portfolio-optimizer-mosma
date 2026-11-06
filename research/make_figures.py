"""Regenerate the backtest figures shown in the README (docs/figures/).

Run from the repository root:  python research/make_figures.py
Results depend on the download date (yfinance) and on the asset universe.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backtest  # noqa: E402
import utils  # noqa: E402

ASSETS = ["ADRO.JK", "AKRA.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK", "BBNI.JK"]
WEALTH = 10_000_000
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

# Validated categorical palette (light surface #fcfcfb)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    data = utils.load_data(tuple(ASSETS), "5y")
    index = utils.load_index("^JKSE", "5y")
    m = data.resample("ME").last()
    im = index.resample("ME").last().reindex(m.index).ffill()

    runs = {}
    for contrarian in (False, True):
        dates, vals, _ = backtest.rolling_mosma_backtest(
            m, im, list(m.columns), WEALTH, 0.0015, 0.0025, 1.0, 12,
            lookback_periods=24, seed=42, contrarian=contrarian)
        runs[contrarian] = (dates, vals)
    dates, vals = runs[False]
    jkse = (im.reindex(dates) / im.reindex(dates).iloc[0] * WEALTH).values

    series = [
        ("Hybride (α corrigé)", vals["Hybrid (MOSMA)"], BLUE, "-", 2.4),
        ("Hybride (α d'origine)", runs[True][1]["Hybrid (MOSMA)"], VIOLET, "-", 1.8),
        ("1/N", vals["Naive (1/N)"], ORANGE, "--", 1.8),
        ("^JKSE", jkse, AQUA, ":", 2.0),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    style(ax)
    ends = []
    for name, v, color, ls, lw in series:
        v = np.asarray(v) / WEALTH * 100
        ax.plot(dates, v, color=color, ls=ls, lw=lw, label=name)
        ends.append([v[-1], name, color])
    ends.sort(key=lambda e: e[0])                       # spread labels so they never overlap
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 4.5)
    for y, name, color in ends:
        ax.annotate(f"{name}", xy=(dates[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.axhline(100, color=MUTED, lw=0.8)
    ax.set_ylabel("Valeur du portefeuille (base 100)", color=MUTED, fontsize=10)
    ax.set_title(f"Backtest walk-forward corrigé – {dates[0]:%m/%Y} → {dates[-1]:%m/%Y}, "
                 "8 actions, frais inclus", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.margins(x=0.01)
    fig.subplots_adjust(right=0.80)
    fig.savefig(os.path.join(OUT, "result_backtest_corrected.png"), dpi=160, facecolor=SURFACE)
    print("saved result_backtest_corrected.png")


if __name__ == "__main__":
    main()
