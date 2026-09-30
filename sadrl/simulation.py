import pandas as pd
import numpy as np
import bokeh
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.layouts import column

from sadrl.broker import Broker

class Simulation:
    def __init__(self, data: pd.DataFrame, cash: float = 100.0) -> None:
        """
        Initialize the Simulation.

        Parameters:
        - data (pd.DataFrame): The financial data for simulation.
        - cash (float): The initial cash balance for the simulation. Default is £100.0.
        """
        # Provided Variables
        self.data = data.reset_index(drop=True)
        self.initial_balance = cash

        # Inferred variables
        self.broker = Broker(self.data, self.initial_balance)

    def step(self, action: int, pos_size: float) -> dict:
        """
        Simulate a single step in the market.

        Parameters:
        - action (int): Action code (1 for Buy, -1 for Sell, 0 for Hold).
        - pos_size (float): Position size for the action.

        Returns:
        - dict: Information about the step including profit, cash, potential return, and direction.
        """
        # Process action in the market
        profit = self.broker.take_action(action, pos_size)

        # Create return dict
        info = {
            "profit": profit, 
            "cash": self.broker.cash, 
            "potential_return": self.broker.potential_return(), 
            "direction": self.broker.current_position.direction if self.broker.current_position is not None else 0
        }

        return info
    
    def results(self):
        """
        Print the simulation results.
        """
        # calmar ratio
        # maximum drawdown

        marginal_returns = [trade.pos_return for trade in self.broker.trades]
        total_return = ((self.broker.cash - self.initial_balance) * 100 )/ self.initial_balance
        risk = np.std(marginal_returns) * 100

        result_dict = {
            "return": total_return, 
            "profit": self.broker.cash - self.initial_balance, 
            "n_trades": len(self.broker.trades), 
            "risk": risk, 
            "sharpe_ratio": total_return / risk
        }
        trades = []
        for trade in self.broker.trades:
            trade_dict = {
                "entry_index": trade.entry_index, 
                "exit_index": trade.exit_index, 
                "entry_price": trade.entry_price, 
                "exit_price": trade.exit_price, 
                "marginal_return": trade.pos_return, 
                "profit": trade.profit
            }
            trades.append(trade_dict)

        trades_df = pd.DataFrame(trades)

        return result_dict, trades_df

    def plot(self, filename):
        """
        Plot the trades of the simulation.
        """

        trades = []
        for trade in self.broker.trades:
            trade_dict = {
                "entry_index": trade.entry_index,
                "exit_index": trade.exit_index,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "marginal_return": trade.pos_return,
                "profit": trade.profit, 
                "max_drawdown": trade.max_drawdown
            }
            trades.append(trade_dict)

        trade_df = pd.DataFrame(trades)

        # Configure source data
        price_dates = self.data[['Close']].index.values
        price_source = ColumnDataSource(data=dict(date=price_dates, price=self.data[['Close']].values))

        # Create figure
        output_file(filename=filename)
        p = figure(height=700, width=1800, tools="pan,wheel_zoom,box_zoom,reset",
                   toolbar_location='right', x_axis_location="below", background_fill_color="#efefef")

        # Plot price
        p.line('date', 'price', source=price_source)

        # Plot trades
        for _, row in trade_df.iterrows():
            x_values = [row['entry_index'], row['exit_index']]
            y_values = [row['entry_price'], row['exit_price']]

            line_col = 'green' if row['profit'] > 0 else 'red'
            p.line(x_values, y_values, line_width=2, line_dash='dashed', line_color=line_col)

        # Formatting
        p.yaxis.axis_label = 'Price'

        # show(column(p))
        print(p)
        save(p)
