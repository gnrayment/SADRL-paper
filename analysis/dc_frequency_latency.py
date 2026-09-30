"""Decision frequency and per-decision compute latency (Table 13, Section 6.1).

frequency: DC-sample one pair's raw ticks over a calendar period at every
           threshold and report event counts and the interval between events.
latency:   time feature construction and one policy forward pass. Compute cost
           depends on the network architecture, not the trained weights, so an
           untrained network of identical architecture (SB3 default TRPO
           MlpPolicy, 26 inputs, 2x64 tanh, 3 actions) is used.

Usage:
  python analysis/dc_frequency_latency.py frequency GBPUSD 2022-01-01 2023-01-01
  python analysis/dc_frequency_latency.py latency
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sadrl.dc_sampling import sample_dc


def frequency(pair, start, end):
    ticks = pd.read_parquet(os.path.join(config.RAW_TICK_DIR, f"{pair}.parquet"), columns=None)
    ticks["Timestamp"] = pd.to_datetime(ticks["Timestamp"])
    ticks = ticks[(ticks.Timestamp >= start) & (ticks.Timestamp < end)].reset_index(drop=True)  # file order, as in the paper
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    rows = []
    for theta in config.THETAS:
        ev = sample_dc(ticks, theta)
        gaps = ev.Timestamp.diff().dt.total_seconds().dropna()
        rows.append({"theta_pct": theta * 100, "events": len(ev), "per_day": len(ev) / days, "mean_s": gaps.mean(),
                     "median_s": gaps.median(), "p10_s": gaps.quantile(.1), "p90_s": gaps.quantile(.9)})
    print(f"{len(ticks)} ticks; median inter-tick gap {ticks.Timestamp.diff().dt.total_seconds().median():.3f} s")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def latency(n=20_000):
    import gym
    import torch
    from sb3_contrib import TRPO

    torch.set_num_threads(1)

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-2, 2, shape=(26,), dtype=np.float32)
        action_space = gym.spaces.Discrete(3)

        def reset(self):
            return np.zeros(26, np.float32)

        def step(self, a):
            return np.zeros(26, np.float32), 0.0, False, {}

    model = TRPO("MlpPolicy", Dummy(), seed=config.SEED)
    obs = np.random.uniform(-2, 2, 26).astype(np.float32)
    for _ in range(2000):
        model.predict(obs, deterministic=True)
    t = []
    for _ in range(n):
        a = time.perf_counter()
        model.predict(obs, deterministic=True)
        t.append(time.perf_counter() - a)
    t = np.array(t) * 1e6

    # Feature update on a new DC event: 20 indicators on a rolling buffer of DCC prices
    buf = np.cumsum(np.random.randn(200)) * 1e-4 + 1.2

    def features(p):
        f = [p[-n:].mean() - p[-n - 1:-1].mean() for n in (10, 20, 30, 40, 50)]
        d = np.diff(p[-31:])
        for n in (14, 21, 30):
            g, l = d[-n:].clip(0).mean(), (-d[-n:]).clip(0).mean()
            f.append(100 - 100 / (1 + g / (l + 1e-12)))
        ema = lambda x, s: pd.Series(x).ewm(span=s, adjust=False).mean().values
        for fa, sl, sg in ((12, 26, 9), (9, 21, 7), (6, 13, 5)):
            macd = ema(p[-60:], fa) - ema(p[-60:], sl)
            sig = ema(macd, sg)
            f += [macd[-1], sig[-1], macd[-1] - sig[-1]]
        mu, sd = p[-14:].mean(), p[-14:].std()
        return np.array(f + [mu + 2 * sd, mu, mu - 2 * sd])

    for _ in range(500):
        features(buf)
    tf = []
    for _ in range(5000):
        a = time.perf_counter()
        features(buf)
        tf.append(time.perf_counter() - a)
    tf = np.array(tf) * 1e6
    print(f"policy inference:     median {np.median(t):.1f} us, p99 {np.percentile(t, 99):.1f} us")
    print(f"feature construction: median {np.median(tf):.1f} us, p99 {np.percentile(tf, 99):.1f} us")


if __name__ == "__main__":
    if sys.argv[1] == "frequency":
        frequency(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        latency()
