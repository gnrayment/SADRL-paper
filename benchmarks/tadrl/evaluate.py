"""Evaluate TADRL on the test blocks of every window for one pair.

Usage: python benchmarks/tadrl/evaluate.py T EURUSD
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import sys
import json
import pandas as pd
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from sadrl.features import standardise
from sadrl.environment import TradingEnv

TADRL_DATA_DIR = os.environ.get('SADRL_TADRL_DATA_DIR', os.path.join(config.ROOT, 'data', 'tadrl'))
TADRL_MODELS_DIR = os.environ.get('SADRL_TADRL_MODELS_DIR', os.path.join(config.ROOT, 'models', 'tadrl'))

# Set seeds
np.random.seed(42)
stable_baselines3.common.utils.set_random_seed(42)

def calculate_metrics(balance_history):
    """Calculate total return, max drawdown, and Calmar ratio."""
    initial_balance = balance_history[0]
    final_balance = balance_history[-1]

    # Total Return
    total_return = (final_balance - initial_balance) / initial_balance * 100  # Convert to percentage

    # Max Drawdown Calculation
    peak_balance = np.maximum.accumulate(balance_history)
    drawdown = (balance_history - peak_balance) / peak_balance
    max_drawdown = np.min(drawdown) * 100  # Most negative drawdown

    # Calmar Ratio
    calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan  # Avoid division by zero

    return total_return, max_drawdown, calmar_ratio  # Convert max drawdown to percentage

def save_results(output_dir, balance_history, total_return, max_drawdown, calmar_ratio):
    """Save results to a JSON file, ensuring the directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "balance_history": balance_history,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio
    }

    output_path = os.path.join(output_dir, "evaluation_results.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")

def evaluate():
    # window = sys.argv[3]
    algo_name = 'PPO'

    freq = sys.argv[1] if len(sys.argv) > 1 else 'T'
    pair = sys.argv[2] if len(sys.argv) > 2 else 'USDCHF'
    for window in range(0, 28, 4):

        try:

            # Load test data
            print('Loading test data...')
            test_df = pd.read_parquet(os.path.join(TADRL_DATA_DIR, freq, pair, f'window_{window}', 'test.parquet'))

            # Standardise test data
            print('Processing test data...')
            # NOTE (kept as run): the test set is standardised with its own statistics rather than the
            # training set's, a small look-ahead that can only favour TADRL.
            _, _, test_set = standardise(test_df, test_df, test_df)

            # Create test environment
            print('Creating test environment...')
            test_env = TradingEnv(test_set, mode='Test')

            # Load trained model
            model_path = os.path.join(TADRL_MODELS_DIR, algo_name, freq, pair, f"window_{window}", "rl_model_1000000_steps.zip")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            print(f'Loading trained model from {model_path}...')
            model = PPO.load(model_path, env=test_env)

            # Run evaluation
            print('Evaluating model on test set...')
            obs = test_env.reset()
            done = False
            balance_history = [test_env.balance]  # Track balance over time

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                balance_history.append(test_env.balance)

            # Compute performance metrics
            total_return, max_drawdown, calmar_ratio = calculate_metrics(balance_history)

            print(f'Total Return: {total_return:.2f}%')
            print(f'Maximum Drawdown: {max_drawdown:.2f}%')
            print(f'Calmar Ratio: {calmar_ratio:.2f}')

            # Define output directory
            output_dir = os.path.join(config.RESULTS_DIR, "benchmarks", "TADRL", algo_name, freq, pair, f"window_{window}")
            
            # Save results to JSON
            save_results(output_dir, balance_history, total_return, max_drawdown, calmar_ratio)

        except:

            print(f"Could not process window {window}")

if __name__ == '__main__':
    evaluate()