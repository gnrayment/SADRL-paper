"""Step 2: train one SADRL agent for a (theta, pair, window, algorithm).

Trains on the training partition only, for a fixed budget of 1,000,000 steps
with 4 parallel environments and library-default hyperparameters. A checkpoint
is saved every 5,000 steps; the final checkpoint is the model that is evaluated.
There is no early stopping and no evaluation callback (Section 5.2).

Usage: python scripts/train.py 0.00015 EURUSD 0 TRPO
"""
import os
import random
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sadrl.environment import TradingEnv
from sadrl.features import load_algorithm, standardise

# Fixed seed for every run (Section 5.2)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
stable_baselines3.common.utils.set_random_seed(config.SEED)


def main(theta, pair, window, algo_name):
    data = os.path.join(config.DATA_DIR, str(theta), pair, f"window_{window}")
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(data, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(data, "test.parquet"))

    # Features are built within each partition; scaling is fitted on training data only.
    print("Processing data...")
    train_set, _, _ = standardise(train_df, val_df, test_df)

    print("Creating environment...")
    env = make_vec_env(lambda: TradingEnv(train_set), n_envs=config.N_ENVS)

    model = load_algorithm(algo_name)("MlpPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=os.path.join(config.MODELS_DIR, algo_name, str(theta), pair, f"window_{window}"),
        verbose=1,
    )

    print("Training model...")
    model.learn(total_timesteps=config.TRAINING_STEPS, callback=checkpoint_callback)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
