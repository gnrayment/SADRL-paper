"""Grid backtests for the MAC and RSI benchmarks (Section 5.3.2, Table 5).

Each configuration is backtested once over the full data period of each pair,
executing at the historical bid and ask and committing the full balance.
For each pair the configuration with the highest total return is reported;
this ex-post selection is disclosed in the paper.

Usage:
  python benchmarks/fixed_interval/run_mac_rsi.py mac
  python benchmarks/fixed_interval/run_mac_rsi.py rsi
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import config
from strategies.MAC_2 import TwoMACrossoverStrategy
from strategies.rsi import RSIStrategy


def total_return(marginal_returns):
    total = 1.0
    for r in marginal_returns.dropna():
        total *= (1 + r)
    return total - 1


def max_drawdown(marginal_returns):
    cum = (1 + marginal_returns.dropna()).cumprod()
    return -((cum - cum.cummax()) / cum.cummax()).min()


def evaluate(strategy):
    strategy.run()
    mr = strategy.get_marginal_returns()
    tr, mdd = total_return(mr), max_drawdown(mr)
    return tr * 100, mdd * 100, (tr / mdd if mdd != 0 else np.nan)


def main(which):
    rows = []
    if which == "mac":
        freqs = ["T", "15T", "H", "4H"]
        combos = [(s, l) for s in (5, 7, 10) for l in (10, 14, 20) if s < l]
    else:
        freqs = ["15T", "H", "4H"]
        combos = [14, 28, 35]
    for pair in config.PAIRS:
        for freq in freqs:
            path = os.path.join(config.BENCHMARK_DATA_DIR, freq, f"{pair}.parquet")
            if not os.path.exists(path):
                print(f"missing {path}")
                continue
            df = pd.read_parquet(path)
            for c in combos:
                if which == "mac":
                    strat = TwoMACrossoverStrategy(data=df, ma_short=c[0], ma_long=c[1], transaction_cost="spread")
                    params = {"ma_short": c[0], "ma_long": c[1]}
                else:
                    strat = RSIStrategy(data=df, rsi_period=c, overbought=70, oversold=30, transaction_cost="spread")
                    params = {"rsi_period": c}
                tr, mdd, cal = evaluate(strat)
                rows.append({"pair": pair, "sample_freq": freq, **params, "total_return": round(tr, 2),
                             "max_drawdown": round(mdd, 2), "calmar_ratio": round(cal, 2) if not np.isnan(cal) else np.nan})
    res = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, "benchmarks", which.upper())
    os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "grid_results.csv"), index=False)
    best = res.loc[res.groupby("pair")["total_return"].idxmax()]   # ex-post selection, as reported
    best.to_csv(os.path.join(out, "best_per_pair.csv"), index=False)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1].lower())
