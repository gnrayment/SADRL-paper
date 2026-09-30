"""Step 1: DC-sample raw tick data and split it into rolling windows.

Reference implementation (see sadrl/dc_sampling.py): the original data
pipeline was not preserved.

Input : <RAW_TICK_DIR>/<PAIR>.parquet with columns Timestamp, Bid, Ask
Output: <DATA_DIR>/<theta>/<PAIR>/window_<w>/{train,val,test}.parquet

Usage: python scripts/prepare_data.py EURUSD 0.00015
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sadrl.dc_sampling import sample_dc, split_windows


def main(pair, theta):
    group = "USD" if "USD" in pair else "NON_USD"
    start, end = config.PERIOD[group]

    ticks = pd.read_parquet(os.path.join(config.RAW_TICK_DIR, f"{pair}.parquet"),
                            columns=None)
    ticks["Timestamp"] = pd.to_datetime(ticks["Timestamp"])
    ticks = ticks[(ticks["Timestamp"] >= start) & (ticks["Timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    # Keep the file's tick order (TrueFX files contain a few out-of-order timestamps; the paper's data
    # were sampled in file order, and re-sorting changes a handful of DC events).
    ticks = ticks.reset_index(drop=True)

    events = sample_dc(ticks, float(theta))
    print(f"{pair} theta={theta}: {len(events)} DC events")

    weeks = (config.WINDOW_WEEKS["train"], config.WINDOW_WEEKS["val"], config.WINDOW_WEEKS["test"])
    for window, parts in split_windows(events, start, config.WINDOWS, weeks):
        out = os.path.join(config.DATA_DIR, str(theta), pair, f"window_{window}")
        os.makedirs(out, exist_ok=True)
        for name, df in parts.items():
            df.to_parquet(os.path.join(out, f"{name}.parquet"))
        print(f"  window {window}: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
