"""TADRL benchmark: the SADRL environment and indicators on fixed-interval (1-minute) data, trained with PPO.

Usage: python benchmarks/tadrl/train.py T EURUSD 0
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import sys
import pandas as pd
import random
import torch
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from sadrl.features import standardise
from sadrl.environment import TradingEnv

TADRL_DATA_DIR = os.environ.get('SADRL_TADRL_DATA_DIR', os.path.join(config.ROOT, 'data', 'tadrl'))
TADRL_MODELS_DIR = os.environ.get('SADRL_TADRL_MODELS_DIR', os.path.join(config.ROOT, 'models', 'tadrl'))

# Set seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
stable_baselines3.common.utils.set_random_seed(42)


def main():

    freq = sys.argv[1]
    pair = sys.argv[2]
    window = sys.argv[3]
    algo_name = 'PPO'

    # Load df
    print('Loading data...')
    train_df = pd.read_parquet(os.path.join(TADRL_DATA_DIR, freq, pair, f'window_{window}', 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(TADRL_DATA_DIR, freq, pair, f'window_{window}', 'val.parquet'))
    test_df = pd.read_parquet(os.path.join(TADRL_DATA_DIR, freq, pair, f'window_{window}', 'test.parquet'))

    # Process training data
    print('Processing data...')
    train_set, _, _ = standardise(train_df, val_df, test_df)

    # Create the environment
    print('Creating environment...')
    env = make_vec_env(lambda: TradingEnv(train_set), n_envs=4)

    # TADRL uses PPO with library-default hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,  # Save the model every 20_000 timesteps per env
        save_path=os.path.join(TADRL_MODELS_DIR, algo_name, freq, pair, f'window_{window}'),
        verbose=1
    )

    print('Training model...')
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

if __name__ == '__main__':
    main()
