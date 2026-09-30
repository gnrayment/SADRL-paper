#!/bin/bash
# Run the full SADRL pipeline for every threshold, pair, window and algorithm.
#
# The paper's 3,136 training runs were executed as independent jobs on a
# university compute cluster. This script runs them sequentially; to use a job
# scheduler, submit each `python ...` line as its own job instead.
#
# Usage: bash scripts/run_experiments.sh
set -e
cd "$(dirname "$0")/.."

THETAS="0.00015 0.00017 0.00019 0.00021 0.00023 0.00025 0.00027 0.00029"
PAIRS="AUDJPY AUDUSD CADJPY CHFJPY EURCHF EURGBP EURJPY EURUSD GBPJPY GBPUSD NZDUSD USDCAD USDCHF USDJPY"
WINDOWS="0 4 8 12 16 20 24"
ALGOS="DQN A2C PPO TRPO"

# 1. DC sampling and windowing
for THETA in $THETAS; do for PAIR in $PAIRS; do
  python scripts/prepare_data.py "$PAIR" "$THETA"
done; done

# 2. Training, and 3. evaluation on the validation and test blocks
for THETA in $THETAS; do for PAIR in $PAIRS; do for W in $WINDOWS; do for ALGO in $ALGOS; do
  python scripts/train.py "$THETA" "$PAIR" "$W" "$ALGO"
  python scripts/evaluate.py "$THETA" "$PAIR" "$W" "$ALGO" val
  python scripts/evaluate.py "$THETA" "$PAIR" "$W" "$ALGO" test
done; done; done; done

# 4. Zero-shot cross-pair evaluation of the TRPO models (within the USD and non-USD groups)
USD="AUDUSD EURUSD GBPUSD NZDUSD USDCAD USDCHF USDJPY"
NONUSD="AUDJPY CADJPY CHFJPY EURCHF EURGBP EURJPY GBPJPY"
for GROUP in "$USD" "$NONUSD"; do
  for THETA in $THETAS; do for SRC in $GROUP; do for TGT in $GROUP; do for W in $WINDOWS; do
    python scripts/evaluate.py "$THETA" "$SRC" "$W" TRPO test --target "$TGT"
  done; done; done; done
done
