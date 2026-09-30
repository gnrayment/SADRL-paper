"""Step 3: evaluate a trained agent and log every trade.

Loads the final checkpoint and trades deterministically through the
validation or test partition. The same trained model is used for validation
and test; there is no retraining. Writes results.json and trades.csv
(one row per completed trade: entry/exit index and price, marginal return,
profit).

Cross-pair (zero-shot) evaluation: pass --target <PAIR> to apply the model
trained on <pair> to the test partition of another pair from the same period
(Section 6.5). Note: the original cross-pair script was not preserved; here the
target pair's data are scaled with the target pair's own training statistics.

Usage:
  python scripts/evaluate.py 0.00015 EURUSD 0 TRPO test
  python scripts/evaluate.py 0.00015 EURUSD 0 TRPO test --target GBPUSD
  python scripts/evaluate.py 0.00015 EURUSD 0 TRPO test --plot
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sadrl.environment import TradingEnv
from sadrl.features import load_algorithm, standardise


def adjust_df(df):
    df["Midprice"] = (df["Bid"] + df["Ask"]) / 2
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("theta")
    ap.add_argument("pair")
    ap.add_argument("window")
    ap.add_argument("algo")
    ap.add_argument("mode", choices=["val", "test"])
    ap.add_argument("--target", help="evaluate on this pair instead (cross-pair)")
    ap.add_argument("--plot", action="store_true", help="also save an HTML trade plot")
    a = ap.parse_args()

    data_pair = a.target or a.pair
    data = os.path.join(config.DATA_DIR, str(a.theta), data_pair, f"window_{a.window}")
    train_df = adjust_df(pd.read_parquet(os.path.join(data, "train.parquet")))
    val_df = adjust_df(pd.read_parquet(os.path.join(data, "val.parquet")))
    test_df = adjust_df(pd.read_parquet(os.path.join(data, "test.parquet")))
    train_df, val_df, test_df = standardise(train_df, val_df, test_df)

    env = TradingEnv(val_df if a.mode == "val" else test_df, mode="Test")

    model_path = os.path.join(config.MODELS_DIR, a.algo, str(a.theta), a.pair,
                              f"window_{a.window}", f"rl_model_{config.TRAINING_STEPS}_steps")
    model = load_algorithm(a.algo).load(model_path)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    metrics, trades = env.simulation.results()
    stage = "cross_pair" if a.target else ("validation" if a.mode == "val" else "test")
    parts = [config.RESULTS_DIR, stage, a.algo, str(a.theta), a.pair]
    if a.target:
        parts.append(a.target)
    out = os.path.join(*parts, f"window_{a.window}", str(config.TRAINING_STEPS), a.mode)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    trades.to_csv(os.path.join(out, "trades.csv"))
    if a.plot:
        env.simulation.plot(filename=os.path.join(out, "trade_plot.html"))
    print(f"Saved {len(trades)} trades to {out}")


if __name__ == "__main__":
    main()
