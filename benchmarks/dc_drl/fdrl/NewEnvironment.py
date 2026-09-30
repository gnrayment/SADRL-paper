import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
import gym

import numpy as np
import pandas as pd
import random
import os
import json
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


class DirectionalChangeEnv(gym.Env):
    def __init__(self, starting_balance, env_config, window_number):

        self.window_number = window_number

        # get data from config
        data = env_config['data']
        asks = env_config['asks']
        bids = env_config['bids']

        self.full_data = data
        self.full_asks = asks
        self.full_bids = bids

        # init state params
        self.context_length = int(params['training']['context_length'])
        self.lag = int(params['training']['lag'])

        # init data
        self.data = self.full_data
        self.asks = self.full_asks
        self.bids = self.full_bids

        # init state
        self.i = self.lag - 1
        self.state = self.data[0:self.i + 1]
        self.ask_price = asks[self.i]
        self.bid_price = bids[self.i]

        # set action and observation space
        self.action_space = gym.spaces.Discrete(2)  # buy, sell
        self.observation_space = gym.spaces.Box(0, 2, shape=self.state.shape)  # DC start, DC end for last 5 timesteps

        # init simulation params
        self.n_prices = len(self.data) - self.lag
        self.balance = starting_balance
        self.entry_price = None
        self.position_size = None
        self.in_position = 0  # -1 for short, 0 for no, 1 for long
        self.trading_log = []

    def step(self, action):

        self.i += 1
        self.n_prices -= 1
        self.state = self.data[self.i - (self.lag-1) : self.i + 1]
        self.ask_price = self.asks[self.i]
        self.bid_price = self.bids[self.i]
        self.mid_price = (self.ask_price + self.bid_price) / 2

        # reward function
        if self.in_position == 0:
            if action == 0:  #  buy
                self.entry_price = self.ask_price  # long at ask price
                self.position_size = self.balance
                self.in_position = 1
                reward = 0
            elif action == 1:  # sell
                self.entry_price = self.bid_price  # short at bid price
                self.position_size = self.balance
                self.in_position = -1
                reward = 0

        elif self.in_position == -1:
            if action == 0:  #  buy
                profit, marginal_return = calculate_profit(self.position_size, self.in_position, 
                                          self.entry_price, self.ask_price)  # exit short position at ask price
                self.trading_log.append({'Trade Index': self.i, 
                                         'Position Size': self.position_size, 
                                         'Trade Type': 'Short', 
                                         'Entry Price': self.entry_price, 
                                         'Exit Price': self.ask_price, 
                                         'Profit': profit, 
                                         'Marginal Return': marginal_return, 
                                         'Window': self.window_number})
                reward = profit
                self.balance += profit
                self.in_position = 0
                self.position_size = None
                self.entry_price = None
            elif action == 1:  # sell
                reward = 0

        elif self.in_position == 1:
            if action == 0:  #  buy
                reward = 0
            elif action == 1:  # sell
                profit, marginal_return = calculate_profit(self.position_size, self.in_position, 
                                          self.entry_price, self.bid_price)  # exit long position as bid price
                self.trading_log.append({'Trade Index': self.i, 
                                         'Position Size': self.position_size, 
                                         'Trade Type': 'Long', 
                                         'Entry Price': self.entry_price, 
                                         'Exit Price': self.bid_price, 
                                         'Profit': profit, 
                                         'Marginal Return': marginal_return, 
                                         'Window': self.window_number})
                reward = profit
                self.balance += profit
                self.in_position = 0
                self.position_size = None
                self.entry_price = None

        if self.n_prices <= 0 or self.balance <= 0:
            done = True
            if self.in_position != 0:
                
                if self.in_position == -1:
                    trade_type = 'Short'
                    exit_price = self.ask_price
                elif self.in_position == 1:
                    trade_type = 'Long'
                    exit_price = self.bid_price

                profit, marginal_return = calculate_profit(self.position_size, self.in_position, 
                                            self.entry_price, exit_price)
                reward = profit

                self.trading_log.append({'Trade Index': self.i, 
                                         'Position Size': self.position_size, 
                                         'Trade Type': trade_type, 
                                         'Entry Price': self.entry_price, 
                                         'Exit Price': exit_price, 
                                         'Profit': profit, 
                                         'Marginal Return': marginal_return, 
                                         'Window': self.window_number})
                self.balance += profit
                self.in_position = 0
        else:
            done = False

        info = {'balance': self.balance, 
                'trading_log': self.trading_log}

        return self.state, reward, done, info
        
    def reset(self, starting_balance):

        # generate data
        self.data = self.full_data
        self.asks = self.full_asks
        self.bids = self.full_bids

        # reset episode variables
        self.i = self.lag - 1
        self.state = self.data[0:self.i + 1]
        self.ask_price = self.asks[self.i]
        self.bid_price = self.bids[self.i]
        self.n_prices = len(self.data) - self.lag
        self.balance = starting_balance
        self.entry_price = None
        self.position_size = None
        self.in_position = 0  # -1 for short, 0 for no, 1 for long
        self.trading_log = []

        return self.state
        
