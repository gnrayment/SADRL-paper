"""Buy-and-hold benchmark (Table 15a).

Enters at the first DC event of the first test block and exits at the last DC
event of the last test block (theta = 0.015%), committing the full balance.

Return is computed relative to the entry price: (p_exit - p_entry) / p_entry.
The original notebook divided by the exit price, which was corrected in the
revised paper. As in the original analysis, the entry price is the bid at the
first event and the exit price the ask at the last event; a real long trade
would enter at the ask and exit at the bid (a difference of roughly one
spread). The original analysis read the exit price from a processed-data
folder labelled window_27; this script uses the last test block of the window
design (window 24), so values may differ slightly from those reported.

Usage: python benchmarks/fixed_interval/buy_and_hold.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def main(theta=0.00015):
    rows = []
    first_w, last_w = config.WINDOWS[0], config.WINDOWS[-1]
    for pair in config.PAIRS:
        d = os.path.join(config.DATA_DIR, str(theta), pair)
        first = pd.read_parquet(os.path.join(d, f"window_{first_w}", "test.parquet"))["Bid"].iloc[0]
        last = pd.read_parquet(os.path.join(d, f"window_{last_w}", "test.parquet"))["Ask"].iloc[-1]
        rows.append({"pair": pair, "total_return": round((last - first) / first * 100, 2)})
    res = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, "benchmarks", "BH")
    os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "buy_and_hold.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
