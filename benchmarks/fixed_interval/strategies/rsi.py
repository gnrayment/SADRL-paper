from .strategy import Strategy

class RSIStrategy(Strategy):
    def __init__(self, data, rsi_period=14, overbought=75, oversold=25, transaction_cost=0.00025):
        """
        Initialize the RSI Strategy.

        Parameters:
        - data (pd.DataFrame): The input data DataFrame.
        - rsi_period (int): The period for calculating RSI.
        - overbought (float): The overbought threshold.
        - oversold (float): The oversold threshold.
        - transaction_cost (float): The transaction cost per trade.
        """
        super().__init__(data)
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.transaction_cost = transaction_cost

    def calculate_RSI(self, series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        RSI = 100 - (100 / (1 + rs))
        return RSI

    def run(self):
        df = self.data.copy()
        
        # Calculate RSI
        df['RSI'] = self.calculate_RSI(df['Midprice'])

        # Generate signals
        df['RSI_shifted'] = df['RSI'].shift(1)

        # Entry and exit signals
        df['buy_signal'] = (df['RSI_shifted'] > self.oversold) & (df['RSI'] <= self.oversold)
        df['sell_signal'] = (df['RSI_shifted'] < self.overbought) & (df['RSI'] >= self.overbought)
        df['exit_buy'] = (df['RSI_shifted'] <= self.oversold) & (df['RSI'] > self.oversold)
        df['exit_sell'] = (df['RSI_shifted'] >= self.overbought) & (df['RSI'] < self.overbought)

        # Initialize position
        df['position'] = 0

        # Position management
        for i in range(1, len(df)):
            if i % 100 == 0:
                print(i)
            if df['buy_signal'].iloc[i]:
                df['position'].iloc[i] = 1
                print('buy')
                # df.loc[i, 'position'] = 1
            elif df['sell_signal'].iloc[i]:
                df['position'].iloc[i] = -1
                print('sell')
                # df.loc[i, 'position'] = -1
            elif df['exit_buy'].iloc[i] and df['position'].iloc[i-1] == 1:
                df['position'].iloc[i] = 0
                # df.loc[i, 'position'] = 0
                print('exit buy')
            elif df['exit_sell'].iloc[i] and df['position'].iloc[i-1] == -1:
                df['position'].iloc[i] = 0
                # df.loc[i, 'position'] = 0
                print('exit sell')
            else:
                df['position'].iloc[i] = df['position'].iloc[i-1]
                # df.loc[i, 'position'] = df['position'].iloc[i-1]
                print('hold')

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
            df['transaction_cost'] = float(self.transaction_cost) * df['abs_position_change']
            df['strategy_returns'] -= df['transaction_cost']

        # Store marginal returns
        self.marginal_returns = df['strategy_returns']