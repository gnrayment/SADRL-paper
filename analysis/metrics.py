"""Performance metrics computed from the per-trade logs.

Conventions (Section 4.6 of the paper, and the original analysis notebook):
  * For each (algorithm, theta, pair), the trades of the seven test (or
    validation) blocks are taken in sequence, giving one 28-week path and one
    value per metric: 112 observations per algorithm.
  * The position force-closed at the end of each block is not counted
    (the last row of each block's trades.csv is dropped).
  * Returns compound: balance_i = balance_{i-1} * (1 + r_i).
  * Total return is computed over the concatenated trades excluding the final
    trade (as in the original notebook); maximum drawdown over all of them.
  * Calmar = total return / maximum drawdown; +inf when the drawdown is zero.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MODES = {"validation": "val", "test": "test"}


def trades_path(stage, algo, theta, pair, window, target=None):
    mode = "val" if stage == "validation" else "test"
    parts = [config.RESULTS_DIR, stage, algo, str(theta), pair]
    if target:
        parts.append(target)
    return os.path.join(*parts, f"window_{window}", str(config.TRAINING_STEPS), mode, "trades.csv")


def load_block(path):
    """Marginal returns of one block, excluding the final forced close. Empty if no trades."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if "marginal_return" not in d or len(d) == 0:
        return d.iloc[0:0] if len(d.columns) else pd.DataFrame(columns=["marginal_return"])
    return d.iloc[:-1]


def load_path(stage, algo, theta, pair, target=None):
    """Concatenated trades over the seven blocks, with a 'block' column (1-7)."""
    blocks = []
    for b, w in enumerate(config.WINDOWS, start=1):
        d = load_block(trades_path(stage, algo, theta, pair, w, target))
        if d is not None and len(d):
            blocks.append(d.assign(block=b))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=["marginal_return", "block"])


def total_return(r):
    r = np.asarray(r, dtype=float)
    return (np.prod(1 + r[:-1]) - 1) * 100 if len(r) else 0.0


def max_drawdown(r):
    r = np.asarray(r, dtype=float)
    if not len(r):
        return 0.0
    c = np.cumprod(1 + r)
    return -(c / np.maximum.accumulate(c) - 1).min() * 100


def calmar(tr, mdd):
    return tr / mdd if mdd > 0 else np.inf


def path_metrics(r):
    tr, mdd = total_return(r), max_drawdown(r)
    return tr, mdd, calmar(tr, mdd)


def metrics_table(stage, algos=config.ALGORITHMS):
    """One row per (algorithm, theta, pair) with total return, max drawdown and Calmar."""
    rows = []
    for algo in algos:
        for theta in config.THETAS:
            for pair in config.PAIRS:
                t = load_path(stage, algo, theta, pair)
                tr, mdd, cal = path_metrics(t["marginal_return"])
                rows.append({"algorithm": algo, "theta": theta, "pair": pair, "total_return": tr,
                             "max_drawdown": mdd, "calmar": cal, "n_trades": len(t)})
    return pd.DataFrame(rows)
