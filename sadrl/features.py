import pandas as pd
import numpy as np
import talib  # TA-Lib: https://ta-lib.org (pip install TA-Lib)

class MinMaxScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.min_val, self.max_val = feature_range

    def fit(self, data):
        self.min_data = np.min(data, axis=0)
        self.max_data = np.max(data, axis=0)

    def transform(self, data):
        scaled_data = (data - self.min_data) / (self.max_data - self.min_data)
        scaled_data = scaled_data * (self.max_val - self.min_val) + self.min_val
        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# create candlestick function
def create_candlesticks(window):
    if len(window) == 2:

        prev_move = window.iloc[0]
        current_move = window.iloc[1]
        indicator_data = current_move.to_dict()

        if current_move['Direction'] == 1:
            ohlc = {
                "High": prev_move['Start'], 
                "Open": prev_move['DCC'], 
                "Low": current_move['Start'], 
                "Close": current_move['DCC']
            }
        elif current_move['Direction'] == -1:
            ohlc = {
                "Low": prev_move['Start'], 
                "Open": prev_move['DCC'], 
                "High": current_move['Start'], 
                "Close": current_move['DCC']
            }

        return {**ohlc, **indicator_data}


def process_set(df):

    # create DC trend ends
    df['End'] = df['Start'].shift(-1)
    df.dropna(inplace=True)
        
    candlestick_data = []
    for window in df.rolling(2):
        formatted_data = create_candlesticks(window)
        if formatted_data:
            candlestick_data.append(formatted_data)

    cs_df = pd.DataFrame(candlestick_data)
    cs_df.reset_index(inplace=True, drop=True)
    cs_df.dropna(inplace=True)

    # Spread
    cs_df['Spread'] = cs_df['Ask'] - cs_df['Bid']

    # Moving average
    periods = [10, 20, 30, 40, 50]
    for period in periods:
        cs_df[f'MA_{period}'] = cs_df['DCC'].rolling(window=period).mean().diff() * 10000

    # RSI
    periods = [14, 21, 30]
    for period in periods:
        cs_df['RSI_' + str(period)] = talib.RSI(cs_df['DCC'], timeperiod=period)
        
    # MACD
    periods = [[12, 26, 9], [9, 21, 7], [6, 13, 5]]
    for params in periods:
        fast_period, slow_period, signal_period = params
        column_prefix = f'MACD_{fast_period}_{slow_period}_{signal_period}'

        macd, signal, histogram = talib.MACD(cs_df['DCC'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        
        cs_df[column_prefix + '_macd'] = macd
        cs_df[column_prefix + '_signal'] = signal
        cs_df[column_prefix + '_histogram'] = histogram
        
    # Bollinger Bands
    periods = [14]
    for period in periods:
        column_prefix = f'BB_{period}'
        upper_band, middle_band, lower_band = talib.BBANDS(cs_df['DCC'], timeperiod=period)

        cs_df[column_prefix + '_upper'] = upper_band
        cs_df[column_prefix + '_middle'] = middle_band
        cs_df[column_prefix + '_lower'] = lower_band

    # Get info and market cols
    info_cols = ['Low', 'Open', 'High', 'Close', 'Bid', 'Ask', 'Timestamp']
    market_cols = [item for item in cs_df.columns if item not in info_cols]

    # process to diffs
    for column in market_cols:
        cs_df[column] = cs_df[column].diff()

    # remove any nans
    cs_df.dropna(inplace=True)

    return cs_df


def create_tick_candlesticks(df, sample_freq="5T"):

    df_ohlc = df["Midprice"].resample(sample_freq).ohlc()

    # Resample Bid and Ask, keeping only the Close value
    df_bid_close = df["Bid"].resample(sample_freq).last()
    df_ask_close = df["Ask"].resample(sample_freq).last()

    # Merge OHLC, Bid_Close, and Ask_Close into one DataFrame
    df_ohlc = df_ohlc.merge(df_bid_close, how="left", left_index=True, right_index=True)
    df_ohlc = df_ohlc.merge(df_ask_close, how="left", left_index=True, right_index=True)

    df_ohlc = df_ohlc.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    return df_ohlc

def process_tick_set(df):

    cs_df = create_tick_candlesticks(df)

    # Spread
    cs_df['Spread'] = cs_df['Ask'] - cs_df['Bid']

    # Moving average
    periods = [10, 20, 30, 40, 50]
    for period in periods:
        cs_df[f'MA_{period}'] = cs_df['Close'].rolling(window=period).mean().diff() * 10000

    # RSI
    periods = [14, 21, 30]
    for period in periods:
        cs_df['RSI_' + str(period)] = talib.RSI(cs_df['Close'], timeperiod=period)
        
    # MACD
    periods = [[12, 26, 9], [9, 21, 7], [6, 13, 5]]
    for params in periods:
        fast_period, slow_period, signal_period = params
        column_prefix = f'MACD_{fast_period}_{slow_period}_{signal_period}'

        macd, signal, histogram = talib.MACD(cs_df['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        
        cs_df[column_prefix + '_macd'] = macd
        cs_df[column_prefix + '_signal'] = signal
        cs_df[column_prefix + '_histogram'] = histogram
        
    # Bollinger Bands
    periods = [14]
    for period in periods:
        column_prefix = f'BB_{period}'
        upper_band, middle_band, lower_band = talib.BBANDS(cs_df['Close'], timeperiod=period)

        cs_df[column_prefix + '_upper'] = upper_band
        cs_df[column_prefix + '_middle'] = middle_band
        cs_df[column_prefix + '_lower'] = lower_band

    # Get info and market cols
    info_cols = ['Low', 'Open', 'High', 'Close', 'Bid', 'Ask', 'Timestamp']
    market_cols = [item for item in cs_df.columns if item not in info_cols]

    # process to diffs
    for column in market_cols:
        cs_df[column] = cs_df[column].diff()

    # remove any nans
    cs_df.dropna(inplace=True)

    return cs_df

def standardise(train_df, val_df, test_df, tick=False):
    
    
    print(train_df.columns)
    print(val_df.columns)
    print()

    # standardise
    if tick:
        train_set, validation_set, test_set = process_tick_set(train_df), process_tick_set(val_df), process_tick_set(test_df)
    else:
        train_set, validation_set, test_set = process_set(train_df), process_set(val_df), process_set(test_df)
    info_cols = ['Low', 'Open', 'High', 'Close', 'Bid', 'Ask', 'Spread', 'Timestamp']
    market_cols = [item for item in train_set.columns if item not in info_cols]
    
    print(train_set.columns)
    print(validation_set.columns)

    for column in market_cols:

        # standardise
        mean_value = train_set[column].mean()
        std_value = train_set[column].std()
        train_set[column] = (train_set[column] - mean_value) / std_value
        validation_set[column] = (validation_set[column] - mean_value) / std_value
        test_set[column] = (test_set[column] - mean_value) / std_value

    # scale between -2 and 2
    scaler = MinMaxScaler(feature_range=(-2, 2))
    scaler.fit(train_set[market_cols])
    train_set[market_cols] = scaler.transform(train_set[market_cols])
    validation_set[market_cols] = scaler.transform(validation_set[market_cols])
    test_set[market_cols] = scaler.transform(test_set[market_cols])

    print(f"Train size: {len(train_set)}")
    print(f"Validation size: {len(validation_set)}")
    print(f"Test size: {len(test_set)}")

    return train_set, validation_set, test_set


def load_algorithm(algo_name):
    """Return the Stable-Baselines3 class for an algorithm name.

    All four algorithms are used with their library-default hyperparameters
    (Stable-Baselines3 / sb3-contrib 1.7.0; see Table 4 of the paper).
    """
    if algo_name == "PPO":
        from stable_baselines3 import PPO
        return PPO
    if algo_name == "A2C":
        from stable_baselines3 import A2C
        return A2C
    if algo_name == "DQN":
        from stable_baselines3 import DQN
        return DQN
    if algo_name == "TRPO":
        from sb3_contrib import TRPO
        return TRPO
    raise ValueError(f"Unknown algorithm: {algo_name}")
