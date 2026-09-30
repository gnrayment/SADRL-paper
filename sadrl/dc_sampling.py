"""Directional-change (DC) sampling and rolling-window splitting.

REFERENCE IMPLEMENTATION. The DC sampling and windowing stage used to produce
the paper's processed data ran in a separate data pipeline that was not
preserved. This module implements Algorithm 1 of the paper and the window
design of Section 5.1. It was checked against the paper's processed data for
GBP/USD at theta = 0.015%: the number of DC events in every test block matches
exactly, apart from the initial indicator warm-up rows that feature
construction discards. Exact agreement at other thresholds and pairs is not
guaranteed.

Output columns, one row per DC confirmation point (DCC), as expected by
sadrl.features.process_set:
    Timestamp  time of the DCC tick
    Start      price at the start point (extreme) of the trend being confirmed
    DCC        mid-price at the DCC tick
    Direction  +1 for an upturn, -1 for a downturn
    Bid, Ask   quotes at the DCC tick (used for execution)
"""
import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # numba is optional but makes sampling ~100x faster
    def njit(f):
        return f


@njit
def _dc_events(price, theta):
    """Algorithm 1. Returns (dcc_index, start_index, direction) arrays."""
    n = len(price)
    dcc = np.empty(n, np.int64)
    start = np.empty(n, np.int64)
    direction = np.empty(n, np.int64)
    k = 0
    upturn = True          # initial event: Upturn
    p_high = price[0]
    p_low = price[0]
    i_high = 0
    i_low = 0
    for t in range(1, n):
        x = price[t]
        if upturn:
            if x <= p_high * (1.0 - theta):          # downturn confirmed
                upturn = False
                dcc[k] = t
                start[k] = i_high
                direction[k] = -1
                k += 1
                p_low = x
                i_low = t
            elif x > p_high:
                p_high = x
                i_high = t
        else:
            if x >= p_low * (1.0 + theta):           # upturn confirmed
                upturn = True
                dcc[k] = t
                start[k] = i_low
                direction[k] = 1
                k += 1
                p_high = x
                i_high = t
            elif x < p_low:
                p_low = x
                i_low = t
    return dcc[:k], start[:k], direction[:k]


def sample_dc(ticks: pd.DataFrame, theta: float) -> pd.DataFrame:
    """DC-sample a tick series.

    ticks: DataFrame with columns Timestamp, Bid, Ask (and optionally Midprice),
           in file order (do not re-sort: the paper's data were sampled in file order).
    theta: threshold as a fraction (0.015% -> 0.00015).
    """
    if "Midprice" in ticks:      # use the data's own mid-price when present
        mid = ticks["Midprice"].to_numpy(dtype=np.float64)
    else:
        mid = ((ticks["Bid"].to_numpy() + ticks["Ask"].to_numpy()) / 2.0).astype(np.float64)
    dcc, start, direction = _dc_events(mid, float(theta))
    return pd.DataFrame({
        "Timestamp": ticks["Timestamp"].to_numpy()[dcc],
        "Start": mid[start],
        "DCC": mid[dcc],
        "Direction": direction,
        "Bid": ticks["Bid"].to_numpy()[dcc],
        "Ask": ticks["Ask"].to_numpy()[dcc],
    })


def split_windows(events: pd.DataFrame, period_start: str, windows=range(0, 28, 4),
                  weeks=(16, 4, 4)):
    """Split DC events into rolling windows of training/validation/test sets.

    Window w covers weeks [w, w + 24) from period_start: 16 training,
    4 validation and 4 test weeks. Events are assigned by DCC timestamp.
    Yields (window, {"train": df, "val": df, "test": df}).
    """
    t0 = pd.Timestamp(period_start)
    ts = pd.to_datetime(events["Timestamp"])
    week = pd.Timedelta(weeks=1)
    n_train, n_val, n_test = weeks
    for w in windows:
        bounds = {
            "train": (t0 + w * week, t0 + (w + n_train) * week),
            "val": (t0 + (w + n_train) * week, t0 + (w + n_train + n_val) * week),
            "test": (t0 + (w + n_train + n_val) * week, t0 + (w + n_train + n_val + n_test) * week),
        }
        yield w, {k: events[(ts >= a) & (ts < b)].reset_index(drop=True) for k, (a, b) in bounds.items()}
