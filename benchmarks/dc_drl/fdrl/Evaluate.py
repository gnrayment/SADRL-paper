import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
"""This script trades the whole dataset using the respective model for each window"""

import gym
from gym import Env
from gym.spaces import Discrete, Box

import numpy as np
import pandas as pd
import sys
import random
import os
import pickle
import torch
import json

from stable_baselines3 import PPO

# Custom imports
from Utils import *
from NewEnvironment import DirectionalChangeEnv


# load experiment parameters
with open(_os.path.join(_HERE, 'params.json'), 'r') as f:
    experiment_params = json.load(f)


class WindowSimulator(object):
    """class to run a single window"""
    def __init__(self, window, starting_balance, experiment_params, mode):

        self.balance = starting_balance
        self.window = window
        self.experiment_params = experiment_params

        # get data
        data_dict = get_data(pair, window, theta)

        if mode == 'Train':
            data = data_dict['train_data']
            asks = data_dict['train_asks']
            bids = data_dict['train_bids']
        elif mode == 'Val':
            data = data_dict['val_data']
            asks = data_dict['val_asks']
            bids = data_dict['val_bids']
        elif mode == 'Test':
            data = data_dict['test_data']
            asks = data_dict['test_asks']
            bids = data_dict['test_bids']

        # generate prices for visualisation later
        self.prices = (asks + bids) / 2

        # load model
        model_path = _os.path.join(_os.environ.get("SADRL_FDRL_MODELS_DIR", _os.path.join(_ROOT, "models", "fdrl")), f"{theta}/PPO/{pair}/PPO/window_{window}")
        self.model = PPO.load(model_path)

        # configure environment
        self.trade_env_config = {
                'data': data, 
                'asks': asks, 
                'bids': bids
                }

    def run_simulation(self):

        # init tracking variables
        total_trades = 0
        trades = []

        # init environment
        self.env = DirectionalChangeEnv(self.balance, self.trade_env_config, self.window)
        obs = self.env.reset(self.balance)
        done = False

        if len(self.trade_env_config['data']) == 0:
            print('No data')
            return self.balance, 0, []

        else:

            # loop over events
            while not done:
                obs = np.array(obs, dtype=np.float64)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                self.balance += reward

            return self.balance, len(info["trading_log"]), info["trading_log"]



if __name__ == "__main__":

    # set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # set script variables
    pair = str(sys.argv[1])
    theta = float(sys.argv[2])
    mode = str(sys.argv[3])
    algo_name = str(sys.argv[4])

    # init simulation values
    balance = 100

    # init logging values
    trades = []
    pair_marginal_returns = []
    result_dict = {'Return (%)': [], 'Risk (%)': [], 'Maximum Drawdown (%)': [], 'Calmar Ratio': [], 'Win Rate (%)': [], 
        'Average Return (%)': [], 'Ave. Positive Returns (%)': []}

    # loop over each window
    for i in range(48):
        window_start_balance = balance

        # create experiment
        try:
            trader = WindowSimulator(i, balance, experiment_params, mode)
            balance, n_trades, trade_details = trader.run_simulation()
        except IndexError:
            n_trades = 0
            trade_details = []
        window_end_balance = balance

        # log trades
        trades.extend(trade_details)
        marginal_returns = [trade['Marginal Return'] for trade in trade_details]
        pair_marginal_returns.extend(marginal_returns)
        window_return = (window_end_balance - window_start_balance) / window_start_balance
        print(f'Window: {i} Balance: {balance} No.Trades: {n_trades}')

        # calculate window metrics
        metrics = Metrics(marginal_returns, window_return)
        result_dict['Return (%)'].append(window_return * 100)
        result_dict['Risk (%)'].append(metrics.risk() * 100)
        result_dict['Maximum Drawdown (%)'].append(metrics.max_drawdown() * 100)
        result_dict['Calmar Ratio'].append(metrics.calmar_ratio())
        result_dict['Win Rate (%)'].append(metrics.win_rate() * 100)
        result_dict['Average Return (%)'].append(metrics.average_return() * 100)
        result_dict['Ave. Positive Returns (%)'].append(metrics.average_pos_returns() * 100)

        # visualise trades
        window_trades_df = pd.DataFrame(trade_details)
        plot_trades(theta, pair, i, window_trades_df, trader.prices)

    # log total metrics
    total_return = (balance - 100) / 100  # calcualte from starting balance of 100
    full_metrics = Metrics(pair_marginal_returns, total_return)
    result_dict['All Trades'] = {}
    result_dict['All Trades']['Return (%)'] = total_return * 100
    result_dict['All Trades']['Risk (%)'] = full_metrics.risk() * 100
    result_dict['All Trades']['Maximum Drawdown (%)'] = full_metrics.max_drawdown() * 100
    result_dict['All Trades']['Calmar Ratio'] = full_metrics.calmar_ratio()
    result_dict['All Trades']['Win Rate (%)'] = full_metrics.win_rate() * 100
    result_dict['All Trades']['Average Return (%)'] = full_metrics.average_return() * 100
    result_dict['All Trades']['Ave. Positive Returns (%)'] = full_metrics.average_pos_returns() * 100

    print(f'{pair} Total Return: {round(total_return * 100, 2)}% Final Balance: £{round(balance, 2)}')

    # log trades
    trades_df = pd.DataFrame(trades)
    trades_dir = f'./Trades/{algo_name}/{theta}/{mode}'
    os.makedirs(trades_dir, exist_ok=True)
    trades_df.to_csv(os.path.join(trades_dir, f'{pair}.csv'))

    # save results
    result_dir = f'./Results/{algo_name}/{theta}/{mode}'
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{pair}.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
