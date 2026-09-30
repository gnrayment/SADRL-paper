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

from collections import Counter

# Custom imports
from Utils import *
from Environment import TestDirectionalChangeEnv


# load experiment parameters
with open(_os.path.join(_HERE, 'params.json'), 'r') as f:
    experiment_params = json.load(f)


class Simulator(object):
    """class to run a single window"""
    def __init__(self, window, balance, experiment_params, mode):

        self.window = window
        self.balance = balance
        self.experiment_params = experiment_params

        # get data
        self.data_dict = get_data(theta, pair, window)

        self.spread_cap = self.data_dict['spread_cap']

        if mode == 'Train':
            data = self.data_dict['train_data']
            asks = self.data_dict['train_asks']
            bids = self.data_dict['train_bids']
        elif mode == 'Val':
            data = self.data_dict['val_data']
            asks = self.data_dict['val_asks']
            bids = self.data_dict['val_bids']
        elif mode == 'Test':
            data = self.data_dict['test_data']
            asks = self.data_dict['test_asks']
            bids = self.data_dict['test_bids']

        # sort data types
        data = data.astype(float)
        asks = asks.astype(float)
        bids = bids.astype(float)

        # check for enough data
        if len(data) == 0:
            self.valid_data = False
            print('No data')
        else:
            self.valid_data = True

        # generate prices for visualisation later
        self.prices = (asks + bids) / 2
        self.asks = asks
        self.bids = bids

        # load model
        self.model = PPO.load(_os.path.join(_os.environ.get("SADRL_PADRL_MODELS_DIR", _os.path.join(_ROOT, "models", "padrl")), f'{theta}/{pair}/window_{window}/PPO'))

        # configure environment
        self.trade_env_config = {
                'data': data, 
                'asks': asks, 
                'bids': bids
                }

        self.spread_cap = self.data_dict['spread_cap']

    def run_simulation(self):

        # init environment
        self.env = TestDirectionalChangeEnv(self.trade_env_config, state_timesteps, self.spread_cap, self.balance)
        obs = self.env.reset(self.balance)
        done = False

        balances = []

        # loop over events
        actions = []
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, done, info = self.env.step(action)
        balance = info['balance']

        # plot equity curve
        equity_curve_dir = f'./EquityCurves/{theta}/{mode}'
        os.makedirs(equity_curve_dir, exist_ok=True)
        plt.plot(balances)
        plt.xlabel('Timestep')
        plt.ylabel('Balance')
        plt.title(f'Equity Curve {pair} @ {theta}')
        plt.savefig(os.path.join(equity_curve_dir, f'{pair}.png'))

        return info["balance"], len(info["trading_log"]), info["trading_log"]

if __name__ == "__main__":

    # set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # set script variables
    theta = float(sys.argv[1])
    pair = str(sys.argv[2])
    mode = str(sys.argv[3])

    context_length = 1000
    state_timesteps = 1
    balance = 100
    start_balance = 100

    # initialise logging all windows
    pair_results = {}
    all_trades = []
    all_marginal_returns = []
    result_dict = {'Return (%)': [], 'Risk (%)': [], 'Sharpe Ratio': [], 'Maximum Drawdown (%)': [], 'Calmar Ratio': [], 'Win Rate (%)': [], 
        'Average Return (%)': [], 'Ave. Positive Returns (%)': []}

    for window in range(28):

        # make sets contiguous
        if window == 0 or window % 4 == 0:

            # initialise logging specific window
            window_result_dict = {
                'Return (%)': None, 'Risk (%)': None, 'Sharpe Ratio': None, 
                'Maximum Drawdown (%)': None, 'Calmar Ratio': None, 'Win Rate (%)': None, 
                'Average Return (%)': None, 'Ave. Positive Returns (%)': None}

            # run simulation
            window_start_balance = balance
            trader = Simulator(window, balance, experiment_params, mode)
            if trader.valid_data is True:
                balance, n_trades, trade_details = trader.run_simulation()
            else:
                n_trades, trade_details = 0, []
            window_end_balance = balance

            # log window trades
            window_trades = trade_details
            window_marginal_returns = [trade['Marginal Return'] for trade in trade_details]
            window_return = (window_end_balance - window_start_balance) / window_start_balance
            print(f'Window: {window} Window Return: {round(window_return * 100, 2)}% Running Balance: {window_end_balance} No.Trades: {n_trades}')

            # calculate window metrics
            window_metrics = Metrics(window_marginal_returns, window_return)
            window_result_dict['Return (%)'] = window_return * 100
            window_result_dict['Risk (%)'] = window_metrics.risk() * 100
            window_result_dict['Sharpe Ratio'] = window_metrics.sharpe_ratio()
            window_result_dict['Maximum Drawdown (%)'] = window_metrics.max_drawdown() * 100
            window_result_dict['Calmar Ratio'] = window_metrics.calmar_ratio()
            window_result_dict['Win Rate (%)'] = window_metrics.win_rate() * 100
            window_result_dict['Average Return (%)'] = window_metrics.average_return() * 100
            window_result_dict['Ave. Positive Returns (%)'] = window_metrics.average_pos_returns() * 100

            pair_results[f"window_{window}"] = window_result_dict

            # log to all trades
            all_trades.extend(window_trades)
            all_marginal_returns.extend(window_marginal_returns)
            all_return = (balance - start_balance) / start_balance

    # calculate all metrics
    all_metrics = Metrics(all_marginal_returns, all_return)
    result_dict['Return (%)'].append(all_return * 100)
    result_dict['Risk (%)'].append(all_metrics.risk() * 100)
    result_dict['Sharpe Ratio'].append(all_metrics.sharpe_ratio())
    result_dict['Maximum Drawdown (%)'].append(all_metrics.max_drawdown() * 100)
    result_dict['Calmar Ratio'].append(all_metrics.calmar_ratio())
    result_dict['Win Rate (%)'].append(all_metrics.win_rate() * 100)
    result_dict['Average Return (%)'].append(all_metrics.average_return() * 100)
    result_dict['Ave. Positive Returns (%)'].append(all_metrics.average_pos_returns() * 100)

    pair_results[f"all_trades"] = result_dict

    # visualise trades
    trades_df = pd.DataFrame(trade_details)
    # plot_trades(theta, pair, mode, trades_df, trader.prices, trader.asks, trader.bids, context_length, state_timesteps)

    # log trades
    # trades_df = pd.DataFrame(all_trades)
    # trades_dir = f'./Trades/{theta}/{context_length}/{state_timesteps}/{mode}'
    # os.makedirs(trades_dir, exist_ok=True)
    # trades_df.to_csv(os.path.join(trades_dir, f'{pair}.csv'))

    # save results
    result_dir = f'./Results_0.55/{theta}/{context_length}/{state_timesteps}/{mode}'
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{pair}.json'), 'w') as f:
        json.dump(pair_results, f, indent=4)
