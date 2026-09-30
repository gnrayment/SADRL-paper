import numpy as np
from .strategy import Strategy

class TwoMACrossoverStrategy(Strategy):
    def __init__(self, data, ma_short=7, ma_long=28, transaction_cost=0.00025):
        """
        Initialize the Two MA Crossover Strategy.

        Parameters:
        - data (pd.DataFrame): The input data DataFrame.
        - ma_short (int): The period for the short-term moving average.
        - ma_long (int): The period for the long-term moving average.
        - transaction_cost (float or str): The transaction cost per trade as a percentage,
          or 'spread' to use bid/ask prices for calculations.
        """
        super().__init__(data)
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.transaction_cost = transaction_cost

        if not (isinstance(transaction_cost, float) or transaction_cost == 'spread'):
            raise ValueError("transaction_cost must be a float or 'spread'")

    def run(self):
        df = self.data.copy()
        
        # Calculate moving averages if not present
        if f'MA_{self.ma_short}' not in df.columns:
            df[f'MA_{self.ma_short}'] = df['Midprice'].rolling(window=self.ma_short).mean()
        if f'MA_{self.ma_long}' not in df.columns:
            df[f'MA_{self.ma_long}'] = df['Midprice'].rolling(window=self.ma_long).mean()

        # Remove rows with NaN values
        df.dropna(subset=[f'MA_{self.ma_short}', f'MA_{self.ma_long}'], inplace=True)

        # Calculate the difference between short and long MAs
        df['MA_diff'] = df[f'MA_{self.ma_short}'] - df[f'MA_{self.ma_long}']

        # Identify crossovers
        df['crossover_up'] = (df['MA_diff'] > 0) & (df['MA_diff'].shift(1) <= 0)
        df['crossover_down'] = (df['MA_diff'] < 0) & (df['MA_diff'].shift(1) >= 0)

        # Set entry and exit signals
        df['entry_signal'] = df['crossover_up']
        df['exit_signal'] = df['crossover_down']

        # Initialize position
        df['position'] = np.nan

        # Set positions based on signals
        df.loc[df['entry_signal'], 'position'] = 1
        df.loc[df['exit_signal'], 'position'] = 0

        # Forward-fill positions
        df['position'] = df['position'].ffill().fillna(0)

        # Calculate returns
        if self.transaction_cost == 'spread':
            # Ensure bid and ask prices are present
            if not {'Bid', 'Ask'}.issubset(df.columns):
                raise ValueError("Bid and Ask columns are required when transaction_cost is 'spread'")
            
            # Calculate price changes using bid prices
            df['price_change'] = df['Bid'].pct_change()
            df['strategy_returns'] = df['position'].shift(1) * df['price_change']

            # Calculate spread percentage
            df['spread_pct'] = (df['Ask'] - df['Bid']) / df['Midprice']

            # Apply transaction costs at trade times
            df['transaction_cost'] = 0.0
            trade_indices = df[df['position'].diff() != 0].index
            df.loc[trade_indices, 'transaction_cost'] = df.loc[trade_indices, 'spread_pct']
            df['strategy_returns'] -= df['transaction_cost']
        else:
            # Transaction cost is a percentage of the position size
            df['price_change'] = df['Midprice'].pct_change()
            df['strategy_returns'] = df['position'].shift(1) * df['price_change']

            # Apply transaction costs
            df['transaction_cost'] = 0.0
            df['abs_position_change'] = df['position'].diff().abs()
            df['transaction_cost'] = self.transaction_cost * df['abs_position_change']
            df['strategy_returns'] -= df['transaction_cost']

        # Store marginal returns
        self.marginal_returns = df['strategy_returns']