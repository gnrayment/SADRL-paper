"""Paths and experiment constants shared by every script.

All locations default to folders inside the repository and can be overridden
with environment variables, e.g.

    export SADRL_DATA_DIR=/path/to/processed/data
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _path(env_var, default):
    return os.environ.get(env_var, os.path.join(ROOT, default))


# Raw TrueFX tick data: one parquet file per pair with columns Timestamp, Bid, Ask.
RAW_TICK_DIR = _path("SADRL_RAW_TICK_DIR", "data/raw")
# DC-sampled, windowed data: <DATA_DIR>/<theta>/<pair>/window_<w>/{train,val,test}.parquet
DATA_DIR = _path("SADRL_DATA_DIR", "data/processed")
# Fixed-interval resampled data for the classical benchmarks: <BENCHMARK_DATA_DIR>/<freq>/<pair>.parquet
BENCHMARK_DATA_DIR = _path("SADRL_BENCHMARK_DATA_DIR", "data/fixed_interval")
# Trained models: <MODELS_DIR>/<algo>/<theta>/<pair>/window_<w>/rl_model_<steps>_steps.zip
MODELS_DIR = _path("SADRL_MODELS_DIR", "models")
# Evaluation outputs (per-trade logs): <RESULTS_DIR>/<stage>/<algo>/<theta>/<pair>/window_<w>/<steps>/<mode>/trades.csv
RESULTS_DIR = _path("SADRL_RESULTS_DIR", "results")

PAIRS = [
    "AUDJPY", "AUDUSD", "CADJPY", "CHFJPY", "EURCHF", "EURGBP", "EURJPY",
    "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]
USD_PAIRS = [p for p in PAIRS if "USD" in p]
NON_USD_PAIRS = [p for p in PAIRS if "USD" not in p]

THETAS = [0.00015, 0.00017, 0.00019, 0.00021, 0.00023, 0.00025, 0.00027, 0.00029]
WINDOWS = list(range(0, 28, 4))          # seven rolling windows, offset by four weeks
ALGORITHMS = ["DQN", "A2C", "PPO", "TRPO"]

TRAINING_STEPS = 1_000_000
N_ENVS = 4
CHECKPOINT_FREQ = 5_000
SEED = 42

# Data periods (Section 5.1 of the paper)
PERIOD = {
    "USD": ("2022-01-01", "2022-12-31"),
    "NON_USD": ("2022-05-01", "2023-04-30"),
}
WINDOW_WEEKS = {"train": 16, "val": 4, "test": 4}
WINDOW_OFFSET_WEEKS = 4
