"""Equal weight portfolio optimization strategy."""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base import OptimizationStrategy, StrategyFactory


class EqualWeightStrategy(OptimizationStrategy):
    """Equal weight portfolio optimization strategy."""
    
    def __init__(self, name: str = "Equal Weight", **kwargs):
        """Initialize equal weight strategy.
        
        Args:
            name: Strategy name
            **kwargs: Additional parameters (not used for equal weight)
        """
        super().__init__(name, **kwargs)
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Optimize portfolio weights using equal weight allocation.
        
        Args:
            returns: DataFrame with date index and ticker columns
            **kwargs: Additional parameters (not used)
            
        Returns:
            Dictionary mapping ticker to equal weight
        """
        # Validate inputs
        self.validate_inputs(returns)
        
        # Get list of tickers
        tickers = list(returns.columns)
        
        if not tickers:
            raise ValueError("No tickers available for optimization")
        
        # Calculate equal weights
        n_assets = len(tickers)
        equal_weight = 1.0 / n_assets
        
        # Create weights dictionary
        weights = {ticker: equal_weight for ticker in tickers}
        
        # Apply constraints if specified
        constraints = kwargs.get('constraints', {})
        if constraints:
            weights = self.apply_constraints(weights, constraints)
        
        return weights
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            'strategy_type': 'equal_weight',
            'description': 'Equal weight allocation across all assets',
            **self.parameters
        }
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} Strategy (Equal Weight)"


# Register the strategy
StrategyFactory.register("equal_weight", EqualWeightStrategy)
