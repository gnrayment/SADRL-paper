import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
import gym

import numpy as np
import pandas as pd
import random
import os
import json
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from Utils import *

# load params
with open(_os.path.join(_HERE, 'params.json'), 'r') as f:
    params = json.load(f)

# set seeds
seed = params['seed']
random.seed(seed)
np.random.seed(seed)

def find_multiplier(number):
    order = (math.floor(math.log10(abs(number))) * -1) - 1
    mulitplier = 10 ** order
    return mulitplier


class TestDirectionalChangeEnv(gym.Env):
    def __init__(self, env_config, state_timesteps, spread_cap, init_balance):

        self.episode_runs = 0

        # get data from config
        data = env_config['data']
        asks = env_config['asks']
        bids = env_config['bids']

        self.full_data = data
        self.full_asks = asks
        self.full_bids = bids

        # init state params
        self.state_timesteps = state_timesteps

        # init data
        self.data = self.full_data
        self.asks = self.full_asks
        self.bids = self.full_bids

        # init simulation params
        self.n_prices = len(self.data) - self.state_timesteps
        self.init_balance = init_balance
        self.balance = self.init_balance
        self.entry_price = None
        self.position_size = None
        self.in_position = 0  # -1 for short, 0 for no, 1 for long
        self.trading_log = []
        self.spread_cap = spread_cap

        # init state
        self.i = self.state_timesteps - 1
        self.ask_price = asks[self.i]
        self.bid_price = bids[self.i]
        self.mid_price = (self.ask_price + self.bid_price) / 2
        self.spread = self.ask_price - self.bid_price
        self.spread_multiplier = find_multiplier(self.spread)
        self.potential_return = 0.0
        sim_variables = [self.in_position, self.spread * self.spread_multiplier, (self.balance-100), self.potential_return * 100]
        self.state = np.append(self.data[0:self.i + 1].flatten(), sim_variables)

        # set action and observation space
        self.action_space = gym.spaces.Discrete(2)  # buy, sell
        self.observation_space = gym.spaces.Box(-2, 2, shape=self.state.shape)  # DC start, DC end for last 5 timesteps
    
    def step(self, action):

        reward = 0
        profit = 0

        # state updates
        starting_balance = self.balance
        self.i += 1
        self.n_prices -= 1
        self.ask_price = self.asks[self.i]
        self.bid_price = self.bids[self.i]
        self.mid_price = (self.ask_price + self.bid_price) / 2
        self.spread = self.ask_price - self.bid_price

        # calcuate potential return
        if self.in_position == 1:  # if long
            p_profit = calculate_profit(self.position_size, self.in_position, self.entry_price, self.bid_price)
            self.potential_return = pct_change(self.balance, self.balance + p_profit)
        elif self.in_position == -1:  # if short
            p_profit = calculate_profit(self.position_size, self.in_position, self.entry_price, self.ask_price)
            self.potential_return = pct_change(self.balance, self.balance + p_profit)
        else:
            self.potential_return = 0.0

        # new state assignment
        sim_variables = [self.in_position, self.spread * self.spread_multiplier, (self.balance-100), self.potential_return * 100]
        self.state = np.append(self.data[self.i - (self.state_timesteps-1) : self.i + 1].flatten(), sim_variables)

        # reward function
        # if self.in_position == 0 and self.spread <= self.spread_cap:
        if self.in_position == 0:
            if action == 0:  #  buy
                self.entry_price = self.ask_price
                self.position_size = self.balance
                self.in_position = 1
            elif action == 1:  # sell
                self.entry_price = self.bid_price
                self.position_size = self.balance
                self.in_position = -1

        elif self.in_position == -1:
            # check potential profit
            if action == 0:  #  buy
                profit = calculate_profit(self.position_size, self.in_position, 
                                          self.entry_price, self.ask_price)
                self.balance += profit

                self.trading_log.append({'Trade Index': self.i, 
                                         'Position Size': self.position_size, 
                                         'Trade Type': 'Short', 
                                         'Entry Price': self.entry_price, 
                                         'Exit Price': self.ask_price, 
                                         'Profit': profit, 
                                         'Marginal Return': pct_change(starting_balance, self.balance)})

                self.in_position = 0
                self.position_size = None
                self.entry_price = None

        elif self.in_position == 1:
            # check potential profit
            if action == 1:  # sell
                profit = calculate_profit(self.position_size, self.in_position, 
                                          self.entry_price, self.bid_price)  # exit long position as bid price

                self.balance += profit
                self.trading_log.append({'Trade Index': self.i, 
                                         'Position Size': self.position_size, 
                                         'Trade Type': 'Long', 
                                         'Entry Price': self.entry_price, 
                                         'Exit Price': self.bid_price, 
                                         'Profit': profit, 
                                         'Marginal Return': pct_change(starting_balance, self.balance)})

                self.in_position = 0
                self.position_size = None
                self.entry_price = None

        if self.n_prices <= 0 or self.balance <= 0:
            done = True
            if self.in_position != 0:
                
                if self.in_position == -1:
                    trade_type = 'Short'
                    exit_price = self.ask_price  # exit short position at ask price
                elif self.in_position == 1:
                    trade_type = 'Long'
                    exit_price = self.bid_price  # exit long position at bid price

                profit = calculate_profit(self.position_size, self.in_position, 
                                            self.entry_price, exit_price)

                self.balance += profit
                self.trading_log.append({'Trade Index': self.i, 
                                         'Position Size': self.position_size, 
                                         'Trade Type': trade_type, 
                                         'Entry Price': self.entry_price, 
                                         'Exit Price': exit_price, 
                                         'Profit': profit, 
                                         'Marginal Return': pct_change(starting_balance, self.balance)})
                self.in_position = 0

            # calculate metrics
            marginal_returns = [trade["Marginal Return"] for trade in self.trading_log]
            risk = np.std(marginal_returns)
            total_return = pct_change(100, self.balance)
            risk_free_rate = 0.0  # 1.5% risk free rate
            try:
                self.sharpe_ratio = total_return / risk
            except:
                self.sharpe_ratio = 0.0

            # calculate final reward
            reward = self.sharpe_ratio

        else:
            done = False

        info = {'balance': self.balance, 
                'trading_log': self.trading_log}
        

        # if reward != 0:
        #     print("---" * 20)
        #     print(f"Total Return: {round(total_return * 100, 2)}%")
        #     print(f"No. Trades: {len(self.trading_log)}")
        #     print(f"Risk: {risk}")
        #     print(f"Sharpe Ratio: {round(self.sharpe_ratio, 2)}")
        #     print(f"Episode: {self.episode_runs}")
        #     print(f"Reward: {reward}")
        #     print("---" * 20)

        return self.state, reward, done, info
        
    def reset(self, starting_balance):

        self.episode_runs += 1

        self.data = self.full_data
        self.asks = self.full_asks
        self.bids = self.full_bids

        # reset episode variables
        self.i = self.state_timesteps - 1
        self.ask_price = self.asks[self.i]
        self.bid_price = self.bids[self.i]
        self.mid_price = (self.ask_price + self.bid_price) / 2
        self.spread = self.ask_price - self.bid_price
        self.n_prices = len(self.data) - self.state_timesteps
        self.balance = starting_balance
        self.entry_price = None
        self.position_size = None
        self.in_position = 0  # -1 for short, 0 for no, 1 for long
        self.trading_log = []
        self.spread = self.ask_price - self.bid_price
        self.potential_return = 0.0
        sim_variables = [self.in_position, self.spread * self.spread_multiplier, (self.balance-100), self.potential_return * 100]
        self.state = np.append(self.data[0:self.i + 1].flatten(), sim_variables)

        return self.state
        

        
