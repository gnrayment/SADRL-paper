import os
import json
import argparse
import numpy as np
import pandas as pd

class MACD_RSI_Strategy:
    def __init__(self, df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9, rsi_period: int = 14, overbought: int = 70, oversold: int = 30):
        self.df = df.copy()
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.df['Mid'] = (self.df['Bid'] + self.df['Ask']) / 2  # Compute mid-price
        self.df['MACD'], self.df['Signal'] = self.calculate_macd(self.df['Mid'])
        self.df['RSI'] = self.calculate_rsi(self.df['Mid'])
        self.df['Position'] = 0
        self.df['TradePrice'] = np.nan
        self.execute_strategy()
    
    def calculate_macd(self, prices):
        short_ema = prices.ewm(span=self.short_window, adjust=False).mean()
        long_ema = prices.ewm(span=self.long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=self.signal_window, adjust=False).mean()
        return macd, signal
    
    def calculate_rsi(self, prices):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def execute_strategy(self):
        position = 0
        for i in range(max(self.long_window, self.rsi_period), len(self.df)):
            macd_above_signal = self.df.iloc[i]['MACD'] > self.df.iloc[i]['Signal']
            macd_below_signal = self.df.iloc[i]['MACD'] < self.df.iloc[i]['Signal']
            rsi_oversold = self.df.iloc[i]['RSI'] < self.oversold
            rsi_overbought = self.df.iloc[i]['RSI'] > self.overbought
            
            if position == 0:
                if macd_above_signal and rsi_oversold:
                    position = 1  # Enter long
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Ask']  # Buy at ask
                elif macd_below_signal and rsi_overbought:
                    position = -1  # Enter short
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Bid']  # Sell at bid
            else:
                if (position == 1 and macd_below_signal) or (position == -1 and macd_above_signal):
                    # Exit: a long position is closed at the bid, a short position at the ask. (v1.0 reset
                    # `position` to 0 before choosing the price, so every exit filled at the ask.)
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Bid'] if position == 1 else self.df.iloc[i]['Ask']
                    position = 0  # Exit trade
            
            self.df.at[self.df.index[i], 'Position'] = position
        
        # Forward-fill trade prices explicitly: pandas < 2.1 did this inside pct_change(), pandas 3 does not.
        self.df['Returns'] = self.df['TradePrice'].ffill().pct_change().fillna(0) * self.df['Position'].shift(1).fillna(0)
        self.df['CumulativeReturns'] = (1 + self.df['Returns']).cumprod()
    
    def get_performance_metrics(self):
        total_return = (self.df['CumulativeReturns'].iloc[-1] - 1) * 100  # Convert to percentage
        running_max = self.df['CumulativeReturns'].cummax()
        drawdown = self.df['CumulativeReturns'] / running_max - 1
        max_drawdown = drawdown.min() * 100  # Convert to percentage
        calmar_ratio = total_return / np.abs(max_drawdown)
        
        return {
            'Total Return (%)': total_return,
            'Maximum Drawdown (%)': max_drawdown,
            'Calmar Ratio': calmar_ratio
        }
    
    def get_results(self):
        return self.df

# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    import config

    parser = argparse.ArgumentParser(description="Run trading strategy and save results.")
    parser.add_argument("pair", type=str, help="Currency pair (e.g., AUDJPY)")
    parser.add_argument("freq", type=str, help="Timeframe (e.g., 4H)")
    args = parser.parse_args()
    
    strategy_name = 'MACDRSI'
    pair = args.pair
    freq = args.freq

    # Load tick data and resample
    path = os.path.join(config.BENCHMARK_DATA_DIR, freq, f"{pair}.parquet")
    df = pd.read_parquet(path, columns=["Bid", "Ask"]).dropna()
    
    strategy = MACD_RSI_Strategy(df)
    metrics = strategy.get_performance_metrics()

    # Define output directory and file path
    output_dir = os.path.join(config.RESULTS_DIR, "benchmarks", strategy_name, freq)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{pair}.json"
    
    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {output_file}")
