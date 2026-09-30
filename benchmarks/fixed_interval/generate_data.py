"""Resample raw ticks to fixed intervals for the classical benchmarks.

Output: <BENCHMARK_DATA_DIR>/<freq>/<PAIR>.parquet (last Bid/Ask/Midprice per bar).
Frequencies used in the paper: T (1 min), 15T, H, 4H.

Usage: python benchmarks/fixed_interval/generate_data.py EURUSD T
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def main(pair, freq):
    df = pd.read_parquet(os.path.join(config.RAW_TICK_DIR, f"{pair}.parquet"), columns=["Timestamp", "Bid", "Ask"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    start, end = config.PERIOD["USD" if "USD" in pair else "NON_USD"]
    df = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].reset_index(drop=True)
    df["Midprice"] = (df["Bid"] + df["Ask"]) / 2
    df = df.set_index("Timestamp").resample(freq).last().dropna(subset=["Midprice"])
    out = os.path.join(config.BENCHMARK_DATA_DIR, freq)
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, f"{pair}.parquet"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
