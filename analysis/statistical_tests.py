"""Friedman tests with Conover post-hoc comparisons (Sections 6.1-6.4).

Post-hoc p-values use the Conover test for the Friedman (blocked) design,
scikit_posthocs.posthoc_conover_friedman, with Holm's step-down correction
applied across all pairwise comparisons. (The original analysis used
posthoc_conover, the test for independent samples; the revised paper reports
the values computed here.) Kendall's W is derived from the Friedman statistic.

Usage:
  # Section 6.1: the four DRL algorithms on the validation sets
  python analysis/statistical_tests.py algorithms

  # Sections 6.2-6.4: SADRL (TRPO, test) against comparators given in a CSV with
  # columns strategy, theta, pair, total_return, max_drawdown, calmar.
  # Leave theta empty for fixed-interval benchmarks (one result per pair; it is
  # repeated across thresholds, as in the paper).
  python analysis/statistical_tests.py strategies --comparators comparators.csv
  python analysis/statistical_tests.py strategies --comparators comparators.csv --pair-level
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from analysis.metrics import metrics_table

METRICS = {"total_return": False, "max_drawdown": True, "calmar": False}   # True = lower is better


def fmt(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.3e}"


def friedman_conover(d, lower_is_better, control):
    """d: rows = blocks, columns = strategies. Returns a summary DataFrame and the Friedman p-value."""
    stat, p = friedmanchisquare(*[d[c] for c in d])
    k, n = d.shape[1], d.shape[0]
    w = stat / (n * (k - 1))   # Kendall's W
    ranks = d.rank(axis=1, ascending=lower_is_better).mean()
    ph = sp.posthoc_conover_friedman(d.reset_index(drop=True), p_adjust="holm")
    out = pd.DataFrame({"avg_rank": ranks.round(2),
                        "p_vs_control": [np.nan if c == control else ph.loc[control, c] for c in d.columns]},
                       index=d.columns).sort_values("avg_rank")
    return out, p, w


def report(title, d, lower, control):
    res, p, w = friedman_conover(d, lower, control)
    print(f"\n== {title}  (n = {len(d)} blocks, Friedman p = {fmt(p)}, Kendall's W = {w:.2f})")
    for s, r in res.iterrows():
        pv = "-" if np.isnan(r.p_vs_control) else fmt(r.p_vs_control)
        print(f"   {s:14s} rank {r.avg_rank:.2f}   p {pv}")


def algorithms():
    t = metrics_table("validation")
    for m, lower in METRICS.items():
        d = t.pivot_table(index=["theta", "pair"], columns="algorithm", values=m)[["TRPO", "PPO", "A2C", "DQN"]]
        report(f"Section 6.1 validation: {m}", d, lower, "TRPO")


def strategies(comparators_csv, pair_level=False):
    sadrl = metrics_table("test", ["TRPO"]).assign(strategy="SADRL").drop(columns=["algorithm", "n_trades"])
    comp = pd.read_csv(comparators_csv)
    dc = comp[comp["theta"].notna()]
    fixed = comp[comp["theta"].isna()].drop(columns="theta")
    fixed = pd.concat([fixed.assign(theta=th) for th in config.THETAS])   # repeat across thresholds
    allres = pd.concat([sadrl, dc, fixed], ignore_index=True)
    for m, lower in METRICS.items():
        d = allres.pivot_table(index=["theta", "pair"], columns="strategy", values=m)
        if pair_level:
            d = d.replace([np.inf, -np.inf], np.nan).groupby(level="pair").mean()
        d = d[["SADRL"] + [c for c in d.columns if c != "SADRL"]]
        report(f"{'Pair-level' if pair_level else 'Pair-threshold'}: {m}", d.dropna(), lower, "SADRL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["algorithms", "strategies"])
    ap.add_argument("--comparators")
    ap.add_argument("--pair-level", action="store_true")
    a = ap.parse_args()
    if a.which == "algorithms":
        algorithms()
    else:
        strategies(a.comparators, a.pair_level)
