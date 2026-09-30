# SADRL: Spread-Aware Deep Reinforcement Learning for FX trading

Code for the paper *A Spread-Aware Deep Reinforcement Learning Framework Using Directional Changes Sampling for High Frequency FX Trading* (G. Rayment, T. Papastylianou, M. Kampouridis).

SADRL trains deep reinforcement learning agents (DQN, A2C, PPO, TRPO) to trade FX on directional-change (DC) sampled tick data. Every trade executes against historical bid and ask quotes, so the agent pays the realised spread during training as well as testing.

## Repository structure

```
config.py                  paths and experiment constants (all paths overridable via environment variables)
sadrl/
  dc_sampling.py           DC sampling (Algorithm 1) and rolling-window split (Section 5.1)  [reference implementation]
  features.py              DC candlesticks, technical indicators, differencing, scaling (Sections 4.1, 4.4)
  environment.py           Gym trading environment: state, actions, reward (Sections 4.3-4.5)
  broker.py, position.py,  execution at bid/ask, full-balance position sizing, trade accounting (Section 4.6)
  simulation.py
scripts/
  prepare_data.py          step 1: DC-sample raw ticks and split into windows
  train.py                 step 2: train one agent (1,000,000 steps, library defaults, seed 42)
  evaluate.py              step 3: evaluate on validation/test blocks, or zero-shot on another pair (--target)
  run_experiments.sh       runs steps 1-3 for all 3,136 configurations and the cross-pair evaluation
benchmarks/
  fixed_interval/          buy-and-hold, MAC, RSI, Mean Reversion, MACD-RSI, Bollinger Bands (Section 5.3.2)
  tadrl/                   TADRL: the SADRL environment on 1-minute bars, trained with PPO
  dc_drl/fdrl, dc_drl/padrl  evaluation of the FDRL and PADRL benchmarks (Section 5.3.1)
analysis/
  metrics.py               total return, maximum drawdown, Calmar ratio from the trade logs
  statistical_tests.py     Friedman + Conover (Friedman design, Holm) tests, pair-level analysis (Sections 6.1-6.4)
  robustness.py            Tables 12, 25-29: risk measures, per-block, slippage, position size, cross-pair, behaviour
  dc_frequency_latency.py  Table 13: DC event frequency and per-decision latency
```

## Installation

The results were produced with Python 3.8.10, Stable-Baselines3 1.7.0, sb3-contrib 1.7.0, PyTorch 1.13.1 and Gym 0.21.

```bash
python3.8 -m venv .venv && source .venv/bin/activate
pip install "setuptools<66" "wheel<0.40"      # needed to install gym 0.21
pip install -r requirements.txt               # TA-Lib needs its C library installed first
```

## Data

Raw tick data (bid and ask quotes) were obtained from [TrueFX](https://www.truefx.com/truefx-historical-downloads/) and are not redistributed here. The periods used are 1 January 2022 to 31 December 2022 for the seven USD pairs and 1 May 2022 to 30 April 2023 for the seven non-USD pairs. Older files may no longer be available from TrueFX; the authors can share the data on reasonable request, subject to TrueFX's terms of use.

Place one parquet file per pair in `data/raw/<PAIR>.parquet` with columns `Timestamp`, `Bid`, `Ask` (and optionally `Midprice`), in the order provided by TrueFX. Paths can be changed in `config.py` or with environment variables (`SADRL_RAW_TICK_DIR`, `SADRL_DATA_DIR`, `SADRL_MODELS_DIR`, `SADRL_RESULTS_DIR`, ...).

## Reproducing the results

```bash
# 1-3. data preparation, training and evaluation (all configurations; long-running)
bash scripts/run_experiments.sh

# or a single configuration
python scripts/prepare_data.py EURUSD 0.00015
python scripts/train.py 0.00015 EURUSD 0 TRPO
python scripts/evaluate.py 0.00015 EURUSD 0 TRPO test

# classical benchmarks
python benchmarks/fixed_interval/generate_data.py EURUSD T      # for each pair and T, 15T, H, 4H
python benchmarks/fixed_interval/run_mac_rsi.py mac
python benchmarks/fixed_interval/run_mac_rsi.py rsi
python benchmarks/fixed_interval/run_rule_based.py
python benchmarks/fixed_interval/buy_and_hold.py

# 4. analysis
python analysis/statistical_tests.py algorithms                 # Tables 7, 9, 11
python analysis/statistical_tests.py strategies --comparators comparators.csv [--pair-level]
python analysis/robustness.py per_block                         # also: slippage, position_size, zero_drawdown,
                                                                #       cross_pair, risk_metrics, behaviour
python analysis/dc_frequency_latency.py frequency GBPUSD 2022-01-01 2023-01-01
python analysis/dc_frequency_latency.py latency
```

Evaluation writes one `trades.csv` per run (every completed trade with its executed entry and exit prices), and all metrics and tests are computed from these logs. The per-trade logs behind the paper (validation, test and cross-pair) are provided as a release asset, so every table in Section 6 can be reproduced without retraining: extract them into `results/` (folders `validation/`, `test/` and `cross_pair/`) and run the analysis scripts.

## Reproducibility notes

- **Training.** Every model is trained on the training partition of its window for a fixed 1,000,000 steps with 4 parallel environments, 100-step episodes from random starting points, and Stable-Baselines3 default hyperparameters (Table 4 of the paper). There is no early stopping; the final checkpoint is evaluated. All runs use seed 42.
- **Position size.** Each trade commits the full current balance (initial balance 100) and returns compound. FDRL and PADRL commit 10% of the balance, as in their original studies.
- **Execution.** An action chosen at one DC confirmation is executed at the bid/ask of the next. Longs open at the ask and close at the bid; shorts open at the bid and close at the ask.
- **Metrics.** For each algorithm, pair and threshold, the seven test blocks are taken in sequence (one 28-week path); the position force-closed at the end of each block is not counted.

## Differences from the original code

This repository is a tidied version of the code that produced the paper's results. The following points are documented for transparency:

- **DC sampling and windowing** (`sadrl/dc_sampling.py`, `scripts/prepare_data.py`) ran in a separate data pipeline that was not preserved. They are re-implemented here from Algorithm 1 and Section 5.1. For GBP/USD at theta = 0.015%, the reference implementation reproduces the number of DC events in every test block of the paper's processed data exactly (after the indicator warm-up rows). Exact agreement elsewhere is not guaranteed.
- **Exploration.** The original training script passed an `epsilon` argument to a custom policy wrapper whose `forward` method Stable-Baselines3 never calls, so it had no effect. It has been removed; each algorithm uses its default exploration.
- **Cross-pair evaluation.** The original script was not preserved; `evaluate.py --target` scales the target pair's data with the target pair's own training statistics.
- **Statistical tests.** Post-hoc p-values use the Conover test for the Friedman design (`posthoc_conover_friedman`) with Holm correction, as reported in the revised paper.
- **Buy-and-hold** is computed relative to the entry price, as reported in the revised paper.
- **Known issues kept so that the reported results reproduce:** the MACD-RSI and Bollinger Bands implementations close long positions at the ask instead of the bid (slightly favouring those benchmarks), and the TADRL evaluation standardises the test data with its own statistics (a small look-ahead favouring TADRL). Both are marked in the code.
- **FDRL and PADRL training** code belongs to the original FDRL and PADRL studies and is not included; only their evaluation code is.

## Citation

If you use this code, please cite the paper (full reference to be added on publication).

## License

MIT (see `LICENSE`).
