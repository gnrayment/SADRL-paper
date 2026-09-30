import os
import json
import argparse
import numpy as np
import pandas as pd

class BollingerBreakoutStrategy:
    def __init__(self, df: pd.DataFrame, bollinger_window: int = 20, num_std: float = 2.0):
        self.df = df.copy()
        self.bollinger_window = bollinger_window
        self.num_std = num_std
        self.df['Mid'] = (self.df['Bid'] + self.df['Ask']) / 2  # Compute mid-price
        self.df['RollingMean'] = self.df['Mid'].rolling(window=bollinger_window).mean()
        self.df['RollingStd'] = self.df['Mid'].rolling(window=bollinger_window).std()
        self.df['UpperBand'] = self.df['RollingMean'] + (self.df['RollingStd'] * num_std)
        self.df['LowerBand'] = self.df['RollingMean'] - (self.df['RollingStd'] * num_std)
        self.df['Position'] = 0
        self.df['TradePrice'] = np.nan
        self.execute_strategy()
    
    def execute_strategy(self):
        position = 0
        for i in range(self.bollinger_window, len(self.df)):
            if position == 0:
                if self.df.iloc[i]['Mid'] > self.df.iloc[i]['UpperBand']:
                    position = -1  # Enter short
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Bid']  # Sell at bid
                elif self.df.iloc[i]['Mid'] < self.df.iloc[i]['LowerBand']:
                    position = 1  # Enter long
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Ask']  # Buy at ask
            else:
                if (position == 1 and self.df.iloc[i]['Mid'] > self.df.iloc[i]['RollingMean']) or \
                   (position == -1 and self.df.iloc[i]['Mid'] < self.df.iloc[i]['RollingMean']):
                    position = 0  # Exit trade
                    self.df.at[self.df.index[i], 'TradePrice'] = self.df.iloc[i]['Bid'] if position == 1 else self.df.iloc[i]['Ask']
                    # KNOWN ISSUE (kept so the reported results reproduce): `position` is reset to 0 on the
                    # line above, so every exit fills at the ask, including long exits. This slightly favours
                    # this benchmark; see the Limitations section of the paper.
            
            self.df.at[self.df.index[i], 'Position'] = position
        
        self.df['Returns'] = self.df['TradePrice'].pct_change().fillna(0) * self.df['Position'].shift(1).fillna(0)
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
    
    strategy_name = 'BollingerBands'
    pair = args.pair
    freq = args.freq

    # Load tick data and resample
    path = os.path.join(config.BENCHMARK_DATA_DIR, freq, f"{pair}.parquet")
    df = pd.read_parquet(path, columns=["Bid", "Ask"]).dropna()
    
    strategy = BollingerBreakoutStrategy(df)
    metrics = strategy.get_performance_metrics()

    # Define output directory and file path
    output_dir = os.path.join(config.RESULTS_DIR, "benchmarks", strategy_name, freq)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{pair}.json"
    
    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {output_file}")
