"""Mean Reversion, MACD-RSI and Bollinger Bands benchmarks (Section 5.3.2).

Runs each strategy with its fixed default parameters at four sampling
intervals for every pair, then reports the interval with the highest total
return per pair (ex-post selection, as disclosed in the paper).

Usage: python benchmarks/fixed_interval/run_rule_based.py
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import config
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.macd_rsi import MACD_RSI_Strategy
from strategies.mean_reversion import MeanReversionStrategy

STRATEGIES = {
    "MeanReversion": (MeanReversionStrategy, False),
    "MACDRSI": (MACD_RSI_Strategy, True),
    "BollingerBands": (BollingerBreakoutStrategy, True),
}
FREQS = ["T", "15T", "H", "4H"]


def main():
    for name, (cls, dropna) in STRATEGIES.items():
        rows = []
        for pair in config.PAIRS:
            for freq in FREQS:
                path = os.path.join(config.BENCHMARK_DATA_DIR, freq, f"{pair}.parquet")
                if not os.path.exists(path):
                    print(f"missing {path}")
                    continue
                df = pd.read_parquet(path, columns=["Bid", "Ask"])
                if dropna:
                    df = df.dropna()
                m = cls(df).get_performance_metrics()
                out = os.path.join(config.RESULTS_DIR, "benchmarks", name, freq)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, f"{pair}.json"), "w") as f:
                    json.dump(m, f, indent=4)
                rows.append({"pair": pair, "sample_freq": freq, **m})
        res = pd.DataFrame(rows)
        best = res.loc[res.groupby("pair")["Total Return (%)"].idxmax()]
        best.to_csv(os.path.join(config.RESULTS_DIR, "benchmarks", name, "best_per_pair.csv"), index=False)
        print(name)
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
