import stable_baselines3 as sb3
import gym
import pandas as pd
import random
import numpy as np

from sadrl.simulation import Simulation

info_cols = [
    'Low', 'Open', 'High', 'Close', 'Bid', 'Ask', 'Spread'
]

market_cols = [
    # 'Start', 'DCC', 
    # 'Direction', 'TMV_1', 
    # 'OSV_1', 'OSV_3', 'OSV_5', 'OSV_10', 
    # 'R_DC_1', 'R_DC_3', 'R_DC_5', 'R_DC_10', 
    # 'T_DC_1', 'T_DC_3', 'T_DC_5', 'T_DC_10', 
    # 'N_DC_1', 'N_DC_10', 'N_DC_20', 'N_DC_30', 'N_DC_40', 'N_DC_50', 
    # 'C_DC_1', 'C_DC_10', 'C_DC_20', 'C_DC_30', 'C_DC_40', 'C_DC_50', 
    'MA_10', 'MA_20', 'MA_30', 'MA_40', 'MA_50', 
    'RSI_14', 'RSI_21', 'RSI_30', 
    'MACD_12_26_9_macd', 'MACD_12_26_9_signal', 'MACD_12_26_9_histogram', 
    'MACD_9_21_7_macd', 'MACD_9_21_7_signal', 'MACD_9_21_7_histogram', 
    'MACD_6_13_5_macd', 'MACD_6_13_5_signal', 'MACD_6_13_5_histogram', 
    'BB_14_upper', 'BB_14_middle', 'BB_14_lower'
]

class TradingEnv(gym.Env):
    def __init__(self, data, mode='Train'):
        super(TradingEnv, self).__init__()

        # persist data and mode
        self.data = data
        self.mode = mode

        # set params of simulation
        if self.mode == 'Train':
            self.max_steps = 100
        elif self.mode == 'Test':
            self.max_steps = len(self.data)

        # organise data into market and info
        self.market_df = data[market_cols]  # get all market data
        self.info_df = data[info_cols]  # get all info data
        position_state_size = 3  # trade direction, equity, current unrealised profit (alter this as appropriate)
        eng_info_size = 3

        # Action space: 0 - Buy, 1 - Sell, 2 - Hold
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: market and positional variables
        state_shape = position_state_size + eng_info_size + self.market_df.shape[-1]
        self.observation_space = gym.spaces.Box(-2, 2, shape=(state_shape,))


    def reset(self):
        """Reset agent at the starting point"""

        # set simulation variables
        self.current_step = 0  # inital value - start one value in to allow for referencing previous values

        # get data sample slices
        if self.mode == 'Train':
            start = random.randint(0, len(self.data) - self.max_steps)  # random start value that is not too close to end
        elif self.mode == 'Test':
            start = 0
        end = start + self.max_steps + 1  # end is self.max_steps from start

        # sample from all data - this will be full data set for test
        self.market_samples = self.market_df.iloc[start:end]  # get slice of market data
        self.info_samples = self.info_df.iloc[start:end]  # get slice of info data

        # initialse trading simulation
        self.simulation = Simulation(self.info_samples)
        
        # create full state
        market_state = self.market_samples.iloc[self.current_step].values  # first row of market_df
        info_state = self.info_samples[['Low', 'Open', 'Close', 'High', 'Spread']].iloc[self.current_step].values
        eng_info = [
            (info_state[3] - info_state[0]) * 1000, 
            (info_state[2] - info_state[1]) * 1000, 
            info_state[4]
        ]
        position_state = np.array([0, 0, 0])  # position, return, current position
        self.state = np.concatenate((position_state, eng_info, market_state))  # concatenate to form a single state

        self.balance = 100

        return self.state

    def step(self, action):
        """Take one step in the envrionment"""

        # Logging
        if self.current_step % 10000 == 0 and self.current_step > 0:
            print(self.current_step)

        # Take action
        results = self.simulation.step(action, self.balance)

        # Calculate reward
        reward = results['profit']

        # Calculate new state
        self.current_step += 1
        self.balance = results['cash']
        if self.current_step < self.max_steps - 1:
            market_state = self.market_samples.iloc[self.current_step].values
            info_state = self.info_samples[['Low', 'Open', 'Close', 'High', 'Spread']].iloc[self.current_step].values
            eng_info = [
                (info_state[3] - info_state[0]) * 1000, 
                (info_state[2] - info_state[1]) * 1000, 
                info_state[4]
            ]
            position_state = np.array([results['direction'], (results['cash']-100)*10, results['potential_return']*1000])
            self.state = np.concatenate((position_state, eng_info, market_state))
            done = False
            return self.state, reward, done, results
        else:
            done = True
            return self.state, reward, done, {"results": results}
            

    def render(self):
        pass
