import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, data):
        """
        Initialize the strategy with the data.

        Parameters:
        - data (pd.DataFrame): The input data DataFrame.
        """
        self.data = data.copy()
        self.marginal_returns = None

    def run(self):
        """
        Run the strategy. Should be implemented by subclasses.
        """
        raise NotImplementedError("Please implement the 'run' method in the subclass.")

    def get_marginal_returns(self):
        """
        Returns the marginal returns from the strategy.

        Returns:
        - marginal_returns (pd.Series): The series of marginal returns.
        """
        if self.marginal_returns is not None:
            return self.marginal_returns
        else:
            raise ValueError("Strategy has not been run yet. Please call the 'run' method first.")