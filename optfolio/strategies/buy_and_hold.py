"""Buy and hold portfolio optimization strategy."""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base import OptimizationStrategy, StrategyFactory


class BuyAndHoldStrategy(OptimizationStrategy):
    """Buy and hold portfolio optimization strategy.
    
    This strategy invests at the start of the backtest and does not rebalance.
    The initial weights are determined by the specified allocation method.
    """
    
    def __init__(self, name: str = "Buy and Hold", allocation_method: str = "equal_weight", **kwargs):
        """Initialize buy and hold strategy.
        
        Args:
            name: Strategy name
            allocation_method: Method for initial allocation ("equal_weight", "market_cap", "custom")
            **kwargs: Additional parameters including custom weights if allocation_method is "custom"
        """
        super().__init__(name, allocation_method=allocation_method, **kwargs)
        self.allocation_method = allocation_method
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Optimize portfolio weights using buy and hold allocation.
        
        Args:
            returns: DataFrame with date index and ticker columns
            **kwargs: Additional parameters including custom weights if needed
            
        Returns:
            Dictionary mapping ticker to initial weight (held throughout backtest)
        """
        # Validate inputs
        self.validate_inputs(returns)
        
        # Get list of tickers
        tickers = list(returns.columns)
        
        if not tickers:
            raise ValueError("No tickers available for optimization")
        
        # Calculate initial weights based on allocation method
        if self.allocation_method == "equal_weight":
            weights = self._calculate_equal_weights(tickers)
        elif self.allocation_method == "market_cap":
            weights = self._calculate_market_cap_weights(tickers, returns, **kwargs)
        elif self.allocation_method == "custom":
            weights = self._get_custom_weights(tickers, **kwargs)
        else:
            raise ValueError(f"Unknown allocation method: {self.allocation_method}")
        
        # Apply constraints if specified
        constraints = kwargs.get('constraints', {})
        if constraints:
            weights = self.apply_constraints(weights, constraints)
        
        return weights
    
    def _calculate_equal_weights(self, tickers: list) -> Dict[str, float]:
        """Calculate equal weights for all tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to equal weight
        """
        n_assets = len(tickers)
        equal_weight = 1.0 / n_assets
        return {ticker: equal_weight for ticker in tickers}
    
    def _calculate_market_cap_weights(self, tickers: list, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Calculate market cap weighted allocation.
        
        This implementation uses a proxy for market cap based on price levels and volatility.
        In practice, you would use actual market cap data from a data provider.
        
        Args:
            tickers: List of ticker symbols
            returns: Returns DataFrame (used for validation)
            **kwargs: Additional parameters including prices if available
            
        Returns:
            Dictionary mapping ticker to market cap weight
        """
        # Try to get prices from kwargs first
        prices = kwargs.get('prices', None)
        
        if prices is not None and not prices.empty:
            # Use price-based proxy for market cap
            # Higher prices and lower volatility suggest larger, more stable companies
            avg_prices = prices.mean()
            volatility = returns.std()
            
            # Create a market cap proxy: higher price / lower volatility = larger market cap
            # Add small constant to avoid division by zero
            market_cap_proxy = avg_prices / (volatility + 1e-6)
            
            # Normalize to weights
            total_proxy = market_cap_proxy.sum()
            weights = (market_cap_proxy / total_proxy).to_dict()
            
            return weights
        else:
            # Fall back to equal weights if no price data available
            print("Warning: No price data available for market cap weighting, falling back to equal weights")
            return self._calculate_equal_weights(tickers)
    
    def _get_custom_weights(self, tickers: list, **kwargs) -> Dict[str, float]:
        """Get custom weights from parameters.
        
        Args:
            tickers: List of ticker symbols
            **kwargs: Additional parameters containing custom_weights
            
        Returns:
            Dictionary mapping ticker to custom weight
            
        Raises:
            ValueError: If custom weights are not provided or invalid
        """
        custom_weights = kwargs.get('custom_weights', {})
        
        if not custom_weights:
            raise ValueError("custom_weights must be provided when allocation_method is 'custom'")
        
        # Validate that all tickers have weights
        missing_tickers = set(tickers) - set(custom_weights.keys())
        if missing_tickers:
            raise ValueError(f"Missing weights for tickers: {missing_tickers}")
        
        # Validate that weights sum to approximately 1
        total_weight = sum(custom_weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Custom weights must sum to 1.0, got {total_weight}")
        
        # Return weights for the tickers in the returns DataFrame
        return {ticker: custom_weights[ticker] for ticker in tickers}
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            'strategy_type': 'buy_and_hold',
            'allocation_method': self.allocation_method,
            'description': f'Buy and hold strategy with {self.allocation_method} initial allocation',
            'rebalancing': False,
            **self.parameters
        }
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} Strategy (Buy and Hold - {self.allocation_method})"


# Register the strategy
StrategyFactory.register("buy_and_hold", BuyAndHoldStrategy)
