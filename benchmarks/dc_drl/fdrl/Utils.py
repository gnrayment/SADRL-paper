import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
import os
import numpy as np
import pandas as pd
import random
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)

# # calculate profit
# def calculate_profit(position_size, trade_direction, entry_price, exit_price):

#     # Entry and exit will be appropriate bid and ask
  
#     price_change = (exit_price - entry_price) / entry_price

#     if trade_direction == 1:
#         profit = price_change * position_size
#     elif trade_direction == -1:
#         profit = -price_change * position_size 
#     else:
#         return 0
#     return profit

# calculate profit
def calculate_profit(position_size, trade_direction, entry_price, exit_price):

    # fixed_t_cost = 0.00025
    size_multiplier = 0.1
    position_size = position_size * size_multiplier
  
    # # fixed transaction cost
    # t_cost = fixed_t_cost * position_size
  
    price_change = (exit_price - entry_price) / entry_price

    if trade_direction == 1:
        profit = price_change * position_size
    elif trade_direction == -1:
        profit = -price_change * position_size
    else:
        return 0, 0

    # trade_profit = profit - t_cost
    trade_profit = profit
    marginal_return = trade_profit / (position_size * entry_price)

    return trade_profit, marginal_return

# get datasets
def get_data(pair, window, theta):

    formatted_theta = '%.5f' % theta


    train_df = pd.read_parquet(_os.path.join(_os.environ.get("SADRL_FDRL_DATA_DIR", _os.path.join(_ROOT, "data", "fdrl")), str(theta), pair, f"window_{window}", "train.parquet"))
    val_df = pd.read_parquet(_os.path.join(_os.environ.get("SADRL_FDRL_DATA_DIR", _os.path.join(_ROOT, "data", "fdrl")), str(theta), pair, f"window_{window}", "val.parquet"))
    test_df = pd.read_parquet(_os.path.join(_os.environ.get("SADRL_FDRL_DATA_DIR", _os.path.join(_ROOT, "data", "fdrl")), str(theta), pair, f"window_{window}", "test.parquet"))

    # get ask prices
    train_asks = train_df['Ask'].values
    val_asks = val_df['Ask'].values
    test_asks = test_df['Ask'].values

    # get bid prices
    train_bids = train_df['Bid'].values
    val_bids = val_df['Bid'].values
    test_bids = test_df['Bid'].values

    # train_df, val_df, test_df = normalize_dataframes(train_df, val_df, test_df)
    train_df, val_df, test_df = manual_normalise(train_df, val_df, test_df)

    # temp remove to match old approach
    # train_df = train_df.drop(['Direction', 'Ask', 'Bid', 'Spread'], axis=1)
    # val_df = val_df.drop(['Direction', 'Ask', 'Bid', 'Spread'], axis=1)
    # test_df = test_df.drop(['Direction', 'Ask', 'Bid', 'Spread'], axis=1)

    train_df = train_df.drop(['Direction', 'Ask', 'Bid'], axis=1)
    val_df = val_df.drop(['Direction', 'Ask', 'Bid'], axis=1)
    test_df = test_df.drop(['Direction', 'Ask', 'Bid'], axis=1)

    if 'JPY' in pair:
        train_df[['Start', 'DCC']] = train_df[['Start', 'DCC']] / 100
        val_df[['Start', 'DCC']] = val_df[['Start', 'DCC']] / 100
        test_df[['Start', 'DCC']] = test_df[['Start', 'DCC']] / 100

    # print(train_df.head)
    # print(train_df.columns.to_list())

    train_data = train_df.values
    val_data = val_df.values
    test_data = test_df.values

    data_dict = {
        'train_data': train_data, 
        'train_asks': train_asks, 
        'train_bids': train_bids,
        'val_data': val_data, 
        'val_asks': val_asks, 
        'val_bids': val_bids,
        'test_data': test_data, 
        'test_asks': test_asks,
        'test_bids': test_bids
    }

    # print("------- Data -------")
    # print(f"Train Shape: {train_data.shape}")
    # print(f"Validation Shape: {val_data.shape}")
    # print(f"Test Shape: {test_data.shape}")

    return data_dict

# percentage change
def pct_change(old_value, new_value):
    change = new_value - old_value
    percentage_change = (change / old_value)
    return percentage_change

# shift by n, new start values are replace by np.nan and end values are discarded
def shift(array, shift):
    return np.concatenate(([np.nan] * shift, array[:-shift]))

# rolling window generator, left over events are discarded
def rolling_window(df, window_size, shift):
    for i in range(0, len(df) - window_size + 1, shift):
        yield df.iloc[i:i+window_size]
        
# take train, validation and test sets and normalise based on training data
def normalize_dataframes(train_df, val_df, test_df):
    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # fit the scaler on the train DataFrame
    scaler.fit(train_df)

    # normalize each DataFrame using the fitted scaler
    train_normalized = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
    val_normalized = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
    test_normalized = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    return train_normalized, val_normalized, test_normalized

def manual_normalise(train_df, val_df, test_df):
    
    DC_start_end = train_df[['Start', 'DCC']]
    train_transformed = (train_df - train_df.min()) / (train_df.max() - train_df.min())
    train_transformed[['Start', 'DCC']] = DC_start_end

    DC_start_end = val_df[['Start', 'DCC']]
    val_transformed = (val_df - train_df.min()) / (train_df.max() - train_df.min())
    val_transformed[['Start', 'DCC']] = DC_start_end

    DC_start_end = test_df[['Start', 'DCC']]
    test_transformed = (test_df - train_df.min()) / (train_df.max() - train_df.min())
    test_transformed[['Start', 'DCC']] = DC_start_end

    return train_transformed, val_transformed, test_transformed

# trade graph
def plot_trades(theta, pair, window, trades_df, prices):

    x = range(len(prices))
    y = prices

    # create a figure and axes
    fig, ax = plt.subplots(figsize=(40, 10))

    # plot the price curve
    line = ax.plot(x, y)

    if len(trades_df) > 0:

        short_index = trades_df[(trades_df['Trade Type'] == 'Short')]['Trade Index'].values
        long_index = trades_df[trades_df['Trade Type'] == 'Long']['Trade Index'].values

        short_trades = [[i, prices[i]] for i in short_index]
        long_trades = [[i, prices[i]] for i in long_index]

        polygons = []
        buys = long_trades
        sells = short_trades

        # trade market dims
        marker_x = 5
        marker_y = (max(prices) - min(prices)) * 0.02

        for buy in buys:
            buy_x, buy_y = buy
            polygons.append(Polygon(xy=[[buy_x, buy_y], [buy_x-marker_x, buy_y-marker_y], [buy_x+marker_x, buy_y-marker_y]], 
                                    closed=True, color='green'))
            
        for sell in sells:
            sell_x, sell_y = sell
            polygons.append(Polygon(xy=[[sell_x, sell_y], [sell_x+marker_x, sell_y+marker_y], [sell_x-marker_x, sell_y+marker_y]], 
                                    closed=True, color='red'))

        for polygon in polygons:
            ax.add_patch(polygon)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Price')
    ax.set_title(f'Window {window}')

    # show the plot
    graph_dir = f'./TradeGraphs/{theta}/{pair}'
    os.makedirs(graph_dir, exist_ok=True)
    plt.savefig(os.path.join(graph_dir, f'Window_{window}.png'))
    plt.close()

# metrics class
class Metrics(object):
    def __init__(self, marginal_returns, total_return):
        self.returns = marginal_returns
        self.total_return = total_return

    def risk(self):
        return np.std(self.returns)

    def max_drawdown(self):
        cumulative_returns = [1]
        [cumulative_returns.append(cumulative_returns[-1] * (1 + r)) for r in self.returns]
        try:
            max_drawdown = max([(max(cumulative_returns[:i+1]) - cumulative_returns[i]) / 
                            max(cumulative_returns[:i+1]) for i in range(1, len(cumulative_returns))])
        except:
            return 0
        return max_drawdown

    def calmar_ratio(self):
        md = self.max_drawdown()
        if md == 0:
            return self.total_return
        else:
            return self.total_return / md

    def win_rate(self):
        positive_returns = [r for r in self.returns if r > 0]
        try:
            win_rate = len(positive_returns) / len(self.returns)
        except:
            return 0
        return win_rate

    def average_return(self):
        try:
            return np.mean(self.returns)
        except:
            return 0

    def average_pos_returns(self):
        try:
            positive_returns = [r for r in self.returns if r > 0]
            return sum(positive_returns) / len(positive_returns)
        except:
            return 0



# trend class
class Trend(object):
    def __init__(self, direction, DC_start, DCC, OS_end, DC_start_index, DCC_index, OS_end_index):
        self.direction, self.DC_start, self.DCC, self.OS_end = direction, DC_start, DCC, OS_end
        self.DC_start_index, self.DCC_index, self.OS_end_index = DC_start_index, DCC_index, OS_end_index
        
        self.data_dict = {
                'Direction': self.direction, 
                'Start': round(self.DC_start, 6),
                'DCC': round(self.DCC, 6),
                'End': round(self.OS_end, 6),
                'Start Index': round(self.DC_start_index, 6),
                'DCC Index': round(self.DCC_index, 6),
                'End Index': round(self.OS_end_index, 6),
            }
        
    def __str__(self):
        return str(self.data_dict)