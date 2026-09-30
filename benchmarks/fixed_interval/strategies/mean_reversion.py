import os
import json
import argparse
import numpy as np
import pandas as pd

class MeanReversionStrategy:
    def __init__(self, df: pd.DataFrame, lookback: int = 20, entry_threshold: float = 0.01, exit_threshold: float = 0.005):
        self.df = df.copy()
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.df['Mid'] = (self.df['Bid'] + self.df['Ask']) / 2  # Compute mid-price
        self.df['RollingMean'] = self.df['Mid'].rolling(window=lookback).mean()
        self.df['RollingStd'] = self.df['Mid'].rolling(window=lookback).std()
        self.df['ZScore'] = (self.df['Mid'] - self.df['RollingMean']) / self.df['RollingStd']
        self.df['Position'] = 0
        self.df['TradePrice'] = np.nan
        self.execute_strategy()
    
    def execute_strategy(self):
        position = 0
        for i in range(self.lookback, len(self.df)):
            if position == 0:
                if self.df.iloc[i]['ZScore'] > self.entry_threshold:
                    position = -1  # Short position
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Bid']  # Short at bid
                elif self.df.iloc[i]['ZScore'] < -self.entry_threshold:
                    position = 1  # Long position
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Ask']  # Long at ask
            else:
                if (position == 1 and self.df.iloc[i]['ZScore'] > -self.exit_threshold):
                    position = 0  # Exit long
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Bid']  # Sell at bid
                elif (position == -1 and self.df.iloc[i]['ZScore'] < self.exit_threshold):
                    position = 0  # Exit short
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Ask']  # Buy at ask
            
            self.df.at[self.df.index[i], 'Position'] = position
        
        self.df['Returns'] = self.df['TradePrice'].pct_change() * self.df['Position'].shift(1)
        self.df['CumulativeReturns'] = (1 + self.df['Returns']).cumprod()
    
    def get_performance_metrics(self):
        total_return = (self.df['CumulativeReturns'].iloc[-1] - 1) * 100  # Convert to percentage
        running_max = self.df['CumulativeReturns'].cummax()
        drawdown = self.df['CumulativeReturns'] / running_max - 1
        max_drawdown = drawdown.min() * 100  # Convert to percentage
        calmar_ratio = total_return / abs(max_drawdown)
        
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
    
    strategy_name = 'MeanReversion'
    pair = args.pair
    freq = args.freq

    # Load tick data and resample
    path = os.path.join(config.BENCHMARK_DATA_DIR, freq, f"{pair}.parquet")
    df = pd.read_parquet(path, columns=["Bid", "Ask"])
    
    strategy = MeanReversionStrategy(df)
    metrics = strategy.get_performance_metrics()

    # Define output directory and file path
    output_dir = os.path.join(config.RESULTS_DIR, "benchmarks", strategy_name, freq)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{pair}.json"
    
    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {output_file}")
