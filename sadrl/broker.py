import pandas as pd

from sadrl.position import Position

class Broker:
    def __init__(self, data: pd.DataFrame, cash: float, drawdown_limit=1.0) -> None:
        """
        Initialize the broker.

        Parameters:
        - data (pd.DataFrame): The data containing OHLC and Bid, Ask.
        - cash (float): Initial cash amount.
        """
        self.data = data
        self.cash = cash
        self.drawdown_limit = drawdown_limit

        self.index = 0
        self.current_position = None
        self.trades = []

    def potential_return(self) -> float:
        """
        Produce the potential returns from exiting current trade.
        """

        # Return 0 if no position
        if self.current_position is None:
            return 0.0
        
        # Get current market information to trade with
        market_state = self.data.iloc[self.index]
        if self.current_position.direction == 1:  # Imaginary exit of Long position
            return self.current_position.marginal_return(market_state['Bid'])
        elif self.current_position.direction == -1:  # Imaginary exit of Short position
            return self.current_position.marginal_return(market_state['Ask'])
        
    def get_max_drawdown(self):
        return (self.equity_peak - self.equity_trough) / self.equity_peak

    def take_action(self, action: int, position_size: float) -> float:
        """
        Execute the specified action.

        Parameters:
        - action (int): Action code (0 for Buy, 1 for Sell, 2 for Hold).
        - position_size (float): The size of the position to buy or sell.

        Returns:
        - float: The profit or loss from the action.
        """

        # Increment position
        self.index += 1

        # Update position max drawdown
        if self.current_position is not None:
            if self.potential_return() < self.current_position.max_drawdown:
                self.current_position.max_drawdown = self.potential_return()

        # Get current market information to trade with
        market_state = self.data.iloc[self.index]

        # Initialise profit
        profit = 0.0

        # Force close on final observation
        if self.index == len(self.data) - 1:
            if self.current_position is None:
                pass
            elif self.current_position.direction == 1:
                profit = self.current_position.close(market_state['Bid'], self.index)
                self.trades.append(self.current_position)
                self.cash += profit
                self.current_position = None
            elif self.current_position.direction == -1:
                profit = self.current_position.close(market_state['Ask'], self.index)
                self.trades.append(self.current_position)
                self.cash += profit
                self.current_position = None

        elif self.index < len(self.data) - 1:

            # Enter trade if currently not in a position
            if self.current_position is None:
                if action == 0:  # Buy
                    self.current_position = Position(market_state['Ask'], self.index, 1, position_size)
                elif action == 1:  # Sell
                    self.current_position = Position(market_state['Bid'], self.index, -1, position_size)
                
            # If in a Long position
            elif self.current_position.direction == 1:
                # Check for drawdown limit
                if self.potential_return() <= -self.drawdown_limit:
                    print('dd limit hit long')
                    profit = self.current_position.close(market_state['Bid'], self.index)
                    self.trades.append(self.current_position)
                    self.cash += profit
                    self.current_position = None
                # Follow action
                elif action == 1:  # Exit
                    profit = self.current_position.close(market_state['Bid'], self.index)
                    self.trades.append(self.current_position)
                    self.cash += profit
                    self.current_position = None

            # If in a Short position
            elif self.current_position.direction == -1:
                # Check for drawdown limit
                if self.potential_return() <= -self.drawdown_limit:
                    print('dd limit hit short')
                    profit = self.current_position.close(market_state['Ask'], self.index)
                    self.trades.append(self.current_position)
                    self.cash += profit
                    self.current_position = None
                # Follow action
                if action == 0:  # Exit
                    profit = self.current_position.close(market_state['Ask'], self.index)
                    self.trades.append(self.current_position)
                    self.cash += profit
                    self.current_position = None

        return profit
