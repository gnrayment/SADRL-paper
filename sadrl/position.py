import pandas as pd

class Position:
    def __init__(self, entry_price: float, entry_index: int, direction: int, position_size: float) -> None:
        """
        Initialize a position.

        Parameters:
        - entry_price (float): The entry price of the position.
        - direction (int): The direction of the position (1 for long, -1 for short).
        - position_size (float): The size of the position.
        """
        if position_size < 0:
            raise ValueError("Position size must be non-negative.")
        
        self.entry_price = entry_price
        self.entry_index = entry_index
        self.direction = direction
        self.position_size = position_size

        self.exit_price = None
        self.exit_index = None
        self.pos_return = None
        self.profit = None

        self.max_drawdown = 0.0

    def marginal_return(self, exit_price: float) -> float:
        """
        Calculate the return and account for direction.

        Parameters:
        - exit_price (float): The exit price of the position.

        Returns:
        - float: The calculated marginal return.
        """
        pct_change = pd.Series([self.entry_price, exit_price]).pct_change().iloc[-1]
        marginal_return = self.direction * pct_change
        return marginal_return

    def close(self, exit_price: float, exit_index: int) -> float:
        """
        Close the position and calculate profit.

        Parameters:
        - exit_price (float): The exit price of the position.

        Returns:
        - float: The calculated profit.
        """
        self.exit_price = exit_price
        self.exit_index = exit_index
        self.pos_return = self.marginal_return(exit_price)
        self.profit = self.pos_return * self.position_size
        return self.profit
