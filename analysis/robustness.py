"""Robustness analyses from the per-trade logs (Sections 6.1, 6.5 and 7.1).

All analyses hold the learned policies fixed and recompute results from the
logged trades; nothing is retrained.

Usage: python analysis/robustness.py <analysis>
  per_block      Table 25: SADRL (TRPO) test performance by test block
  slippage       Table 26: additional execution cost of c pips per side
  position_size  Table 27: allocation of 10/25/50/100% of the balance
  zero_drawdown  Section 6.5: sensitivity of the 6.1 Calmar comparison to undefined ratios
  cross_pair     Table 28: zero-shot evaluation on other pairs
  risk_metrics   Table 12: supplementary risk measures for the four algorithms
  behaviour      Table 29: trading behaviour of SADRL on the test set
Outputs are written to <RESULTS_DIR>/analysis/.
"""
import os
import sys

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from analysis.metrics import load_block, load_path, metrics_table, path_metrics, trades_path

OUT = os.path.join(config.RESULTS_DIR, "analysis")


def save(df, name):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name))
    print(df.round(3).to_string())


def block_rows(stage, algo):
    """Per-block metrics: one row per (theta, pair, block)."""
    rows = []
    for theta in config.THETAS:
        for pair in config.PAIRS:
            for b, w in enumerate(config.WINDOWS, start=1):
                d = load_block(trades_path(stage, algo, theta, pair, w))
                if d is None:
                    continue
                r = d["marginal_return"].to_numpy(dtype=float) if len(d) else np.array([])
                if len(r):
                    c = np.cumprod(1 + r)
                    tr = (c[-1] - 1) * 100
                    eq = np.concatenate([[1.0], c])
                    mdd = -(eq / np.maximum.accumulate(eq) - 1).min() * 100
                else:
                    tr = mdd = 0.0
                rows.append({"theta": theta, "pair": pair, "block": b, "n_trades": len(r), "total_return": tr,
                             "max_drawdown": mdd, "win_rate": (r > 0).mean() * 100 if len(r) else np.nan})
    return pd.DataFrame(rows)


def per_block():
    t = block_rows("test", "TRPO")

    def s(g):
        return pd.Series({"mean_return": g.total_return.mean(), "median_return": g.total_return.median(),
                          "pct_positive": (g.total_return > 0).mean() * 100, "mean_mdd": g.max_drawdown.mean(),
                          "pct_zero_dd": (g.max_drawdown == 0).mean() * 100, "median_trades": g.n_trades.median()})
    res = t.groupby("block").apply(s)
    res.loc["All"] = s(t)
    save(res, "per_block.csv")


def with_direction(t):
    dp = np.sign(t.exit_price - t.entry_price)
    d = (np.sign(t.marginal_return) * dp).replace(0, np.nan)
    return t.assign(direction=d.fillna(1.0))


def slippage(costs=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)):
    paths = {(th, p): with_direction(load_path("test", "TRPO", th, p)) for th in config.THETAS for p in config.PAIRS}

    def metrics(c):
        out = []
        for (th, p), t in paths.items():
            s = c * (0.01 if "JPY" in p else 0.0001)
            e = t.entry_price + t.direction * s
            x = t.exit_price - t.direction * s
            out.append(path_metrics(t.direction * (x - e) / e))
        return pd.DataFrame(out, columns=["tr", "mdd", "cal"])

    rows = {}
    for c in costs:
        m = metrics(c)
        rows[c] = {"mean_return": m.tr.mean(), "median_return": m.tr.median(), "pct_positive": (m.tr > 0).mean() * 100,
                   "mean_mdd": m.mdd.mean(), "mean_calmar": m.cal.replace(np.inf, np.nan).mean(),
                   "median_calmar": m.cal.median()}
    lo, hi = 0.0, 5.0
    for _ in range(40):                      # break-even cost: mean return = 0
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if metrics(mid).tr.mean() > 0 else (lo, mid)
    print(f"break-even additional cost: {lo:.3f} pips per side")
    save(pd.DataFrame(rows).T.rename_axis("pips_per_side"), "slippage.csv")


def position_size(fractions=(0.1, 0.25, 0.5, 1.0)):
    paths = {(th, p): load_path("test", "TRPO", th, p)["marginal_return"].to_numpy(dtype=float)
             for th in config.THETAS for p in config.PAIRS}
    rows = {}
    for f in fractions:
        m = pd.DataFrame([path_metrics(f * r) for r in paths.values()], columns=["tr", "mdd", "cal"])
        rows[f] = {"mean_return": m.tr.mean(), "mean_mdd": m.mdd.mean(), "mean_calmar": m.cal.replace(np.inf, np.nan).mean(),
                   "median_calmar": m.cal.median(), "pct_positive": (m.tr > 0).mean() * 100}
    save(pd.DataFrame(rows).T.rename_axis("allocation"), "position_size.csv")


def zero_drawdown():
    t = metrics_table("validation")
    tr = t.pivot_table(index=["theta", "pair"], columns="algorithm", values="total_return")[["TRPO", "PPO", "A2C", "DQN"]]
    md = t.pivot_table(index=["theta", "pair"], columns="algorithm", values="max_drawdown")[["TRPO", "PPO", "A2C", "DQN"]]
    variants = {"as published (zero drawdown = +inf)": (tr / md.replace(0, np.nan)).fillna(np.inf),
                "zero-drawdown combinations excluded": (tr / md)[(md > 0).all(axis=1)],
                "drawdown floored at 0.01%": tr / md.clip(lower=0.01)}
    print("zero-drawdown cells:", (md == 0).sum().to_dict())
    for name, d in variants.items():
        p = friedmanchisquare(*[d[c] for c in d]).pvalue
        ph = sp.posthoc_conover_friedman(d.reset_index(drop=True), p_adjust="holm")["TRPO"]
        rk = d.rank(axis=1, ascending=False).mean()
        print(f"{name}: n={len(d)} Friedman p={p:.3e} | " +
              " ".join(f"{c} {rk[c]:.2f}/{'-' if c == 'TRPO' else f'{ph[c]:.3f}'}" for c in d))


def cross_pair():
    rows = []
    groups = [config.USD_PAIRS, config.NON_USD_PAIRS]
    for grp in groups:
        for src in grp:
            for tgt in grp:
                for th in config.THETAS:
                    rets = []
                    for w in config.WINDOWS:
                        d = load_block(trades_path("cross_pair", "TRPO", th, src, w, target=tgt))
                        if d is not None and len(d):
                            rets.append((np.prod(1 + d["marginal_return"].to_numpy(dtype=float)) - 1))
                        elif d is not None:
                            rets.append(0.0)
                    if rets:
                        rows.append({"theta": th, "source": src, "target": tgt,
                                     "total_return": (np.prod(1 + np.array(rets)) - 1) * 100})
    c = pd.DataFrame(rows)
    c["case"] = np.where(c.source == c.target, "own pair",
                np.where(c.source.str.contains("JPY") == c.target.str.contains("JPY"), "unseen, same price scale",
                         "unseen, JPY <-> non-JPY"))
    res = c.groupby("case").total_return.agg(cases="count", mean="mean", median="median",
                                             pct_positive=lambda s: (s > 0).mean() * 100)
    save(res, "cross_pair.csv")


def risk_metrics():
    blocks = {a: block_rows("test", a) for a in config.ALGORITHMS}
    res = {}
    for a in config.ALGORITHMS:
        cells, pooled = [], []
        for th in config.THETAS:
            for p in config.PAIRS:
                r = load_path("test", a, th, p)["marginal_return"].to_numpy(dtype=float)
                pooled.append(r)
                b = blocks[a][(blocks[a].theta == th) & (blocks[a].pair == p)].sort_values("block").total_return / 100
                sd = b.std()
                cells.append({
                    "sharpe": b.mean() / sd * np.sqrt(13) if sd > 0 else np.nan,
                    "no_losing_block": b.min() >= 0,
                    "profit_factor": r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.nan,
                    "hit_rate": (r > 0).mean() * 100 if len(r) else np.nan,
                    "avg_trade": r.mean() * 100 if len(r) else np.nan,
                    "downside_dev": np.sqrt((np.minimum(r, 0) ** 2).mean()) * 100 if len(r) else np.nan,
                    "trades_per_block": len(r) / max(len(b), 1)})
        c = pd.DataFrame(cells)
        allr = np.concatenate(pooled) * 100
        var = -np.percentile(allr, 5)
        res[a] = {"sharpe_median": c.sharpe.median(), "sharpe_positive_pct": (c.sharpe > 0).mean() * 100,
                  "no_losing_block_pct": c.no_losing_block.mean() * 100, "profit_factor_median": c.profit_factor.median(),
                  "hit_rate_median": c.hit_rate.median(), "avg_trade_median": c.avg_trade.median(),
                  "downside_dev_median": c.downside_dev.median(), "var95": var,
                  "es95": -allr[allr <= -var].mean(), "trades_per_block_median": c.trades_per_block.median()}
    save(pd.DataFrame(res), "risk_metrics.csv")


def behaviour():
    parts = []
    for th in config.THETAS:
        for p in config.PAIRS:
            t = load_path("test", "TRPO", th, p)
            if len(t):
                dp = np.sign(t.exit_price - t.entry_price)
                d = (np.sign(t.marginal_return) * dp).replace({1: "long", -1: "short", 0: np.nan})
                parts.append(t.assign(direction=d, hold=t.exit_index - t.entry_index, pair=p, theta=th))
    T = pd.concat(parts, ignore_index=True)

    def s(g):
        r = g.marginal_return * 100
        w, l = r[r > 0], r[r < 0]
        return pd.Series({"trades": len(g), "win_rate": (r > 0).mean() * 100, "avg_win": w.mean(), "avg_loss": l.mean(),
                          "payoff_ratio": w.mean() / -l.mean(), "profit_factor": w.sum() / -l.sum(),
                          "avg_trade": r.mean(), "median_hold_events": g.hold.median(),
                          "hold_q25": g.hold.quantile(.25), "hold_q75": g.hold.quantile(.75)})
    res = pd.DataFrame({"all": s(T), "long": s(T[T.direction == "long"]), "short": s(T[T.direction == "short"])})
    save(res, "behaviour.csv")


if __name__ == "__main__":
    {"per_block": per_block, "slippage": slippage, "position_size": position_size, "zero_drawdown": zero_drawdown,
     "cross_pair": cross_pair, "risk_metrics": risk_metrics, "behaviour": behaviour}[sys.argv[1]]()
