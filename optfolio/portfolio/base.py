"""Base portfolio class for portfolio management."""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..portfolio.metrics import PortfolioMetrics


class Portfolio:
    """Base portfolio class for managing portfolio state and rebalancing."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.02,
                 transaction_costs: float = 0.001):
        """Initialize portfolio.
        
        Args:
            initial_capital: Initial portfolio value
            risk_free_rate: Annual risk-free rate
            transaction_costs: Fixed transaction cost per trade (in dollars)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs
        
        # Portfolio state
        self.weights: Dict[str, float] = {}
        self.positions: Dict[str, float] = {}
        self.prices: Dict[str, float] = {}
        
        # Historical data
        self.portfolio_values: List[float] = []
        self.portfolio_dates: List[datetime] = []
        self.weight_history: List[Dict[str, float]] = []
        self.transaction_history: List[Dict] = []
        
        # Metrics calculator
        self.metrics = PortfolioMetrics(risk_free_rate)
        
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set portfolio weights.
        
        Args:
            weights: Dictionary mapping ticker to weight
        """
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
        self.weights = weights.copy()
        
    def get_weights(self) -> Dict[str, float]:
        """Get current portfolio weights.
        
        Returns:
            Dictionary of current weights
        """
        return self.weights.copy()
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update asset prices.
        
        Args:
            prices: Dictionary mapping ticker to current price
        """
        self.prices.update(prices)
        
    def calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value.
        
        Returns:
            Current portfolio value
        """
        if not self.positions or not self.prices:
            return self.current_capital
            
        total_value = 0.0
        for ticker, shares in self.positions.items():
            if ticker in self.prices:
                total_value += shares * self.prices[ticker]
                
        return total_value
    
    def rebalance(self, target_weights: Dict[str, float], 
                 current_prices: Dict[str, float]) -> Dict[str, float]:
        """Rebalance portfolio to target weights.
        
        Args:
            target_weights: Target weights for each asset
            current_prices: Current prices for each asset
            
        Returns:
            Dictionary of trades executed (ticker: shares)
        """
        # Update prices
        self.update_prices(current_prices)
        
        # Calculate current portfolio value
        current_value = self.calculate_portfolio_value()
        
        # Calculate target positions
        target_positions = {}
        for ticker, weight in target_weights.items():
            if ticker in current_prices and current_prices[ticker] > 0:
                target_shares = (weight * current_value) / current_prices[ticker]
                target_positions[ticker] = target_shares
                
        # Calculate trades needed
        trades = {}
        total_transaction_cost = 0.0
        
        for ticker, target_shares in target_positions.items():
            current_shares = self.positions.get(ticker, 0.0)
            trade_shares = target_shares - current_shares
            
            if abs(trade_shares) > 1e-6:  # Only trade if significant difference
                trades[ticker] = trade_shares
                
                # Calculate transaction costs (fixed cost per transaction)
                transaction_cost = self.transaction_costs
                total_transaction_cost += transaction_cost
                
        # Execute trades
        for ticker, shares in trades.items():
            current_shares = self.positions.get(ticker, 0.0)
            self.positions[ticker] = current_shares + shares
            
        # Apply transaction costs by reducing all positions proportionally
        if total_transaction_cost > 0:
            # Calculate the reduction factor to account for transaction costs
            post_trade_value = self.calculate_portfolio_value()
            reduction_factor = (post_trade_value - total_transaction_cost) / post_trade_value
            
            # Apply the reduction to all positions
            for ticker in self.positions:
                self.positions[ticker] *= reduction_factor
            
            # Update current capital to reflect the transaction cost impact
            self.current_capital = self.calculate_portfolio_value()
        else:
            self.current_capital = self.calculate_portfolio_value()
        
        # Record transaction
        if trades:
            self.transaction_history.append({
                'date': datetime.now(),
                'trades': trades.copy(),
                'transaction_cost': total_transaction_cost,
                'portfolio_value': self.current_capital
            })
            
        # Update weights
        self.set_weights(target_weights)
        
        return trades
    
    def step(self, date: datetime, prices: Dict[str, float]) -> float:
        """Step portfolio forward one period.
        
        Args:
            date: Current date
            prices: Current prices for all assets
            
        Returns:
            Current portfolio value
        """
        # Update prices
        self.update_prices(prices)
        
        # Calculate new portfolio value
        portfolio_value = self.calculate_portfolio_value()
        
        # Record portfolio state
        self.portfolio_values.append(portfolio_value)
        self.portfolio_dates.append(date)
        self.weight_history.append(self.weights.copy())
        
        return portfolio_value
    
    def get_portfolio_series(self) -> pd.Series:
        """Get portfolio value time series.
        
        Returns:
            Series with dates as index and portfolio values
        """
        if not self.portfolio_dates:
            return pd.Series([self.initial_capital], index=[datetime.now()])
        
        # Ensure lengths match
        if len(self.portfolio_values) != len(self.portfolio_dates):
            # If there's a mismatch, truncate to the shorter length
            min_length = min(len(self.portfolio_values), len(self.portfolio_dates))
            return pd.Series(
                self.portfolio_values[:min_length], 
                index=self.portfolio_dates[:min_length]
            )
        
        # Create series and ensure no duplicate indices
        series = pd.Series(self.portfolio_values, index=self.portfolio_dates)
        if series.index.duplicated().any():
            # If there are duplicates, keep the last occurrence
            series = series[~series.index.duplicated(keep='last')]
            
        return series
    
    def get_weight_history(self) -> pd.DataFrame:
        """Get historical weight data.
        
        Returns:
            DataFrame with dates as index and weights as columns
        """
        if not self.weight_history:
            return pd.DataFrame()
            
        return pd.DataFrame(self.weight_history, index=self.portfolio_dates)
    
    def get_performance_metrics(self, 
                              market_returns: Optional[pd.Series] = None,
                              benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate portfolio performance metrics.
        
        Args:
            market_returns: Market returns for beta calculation
            benchmark_returns: Benchmark returns for information ratio
            
        Returns:
            Dictionary of performance metrics
        """
        portfolio_series = self.get_portfolio_series()
        return self.metrics.calculate_all_metrics(
            portfolio_series, 
            market_returns, 
            benchmark_returns
        )
    
    def get_rolling_metrics(self, 
                          window: int = 252,
                          market_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate rolling performance metrics.
        
        Args:
            window: Rolling window size
            market_returns: Market returns for beta calculation
            
        Returns:
            DataFrame with rolling metrics
        """
        portfolio_series = self.get_portfolio_series()
        return self.metrics.calculate_rolling_metrics(
            portfolio_series, 
            window, 
            market_returns
        )
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.current_capital = self.initial_capital
        self.weights = {}
        self.positions = {}
        self.prices = {}
        self.portfolio_values = []
        self.portfolio_dates = []
        self.weight_history = []
        self.transaction_history = []
    
    def get_summary(self) -> Dict:
        """Get portfolio summary.
        
        Returns:
            Dictionary with portfolio summary information
        """
        portfolio_series = self.get_portfolio_series()
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital) - 1,
            'current_weights': self.weights.copy(),
            'current_positions': self.positions.copy(),
            'num_transactions': len(self.transaction_history),
            'total_transaction_costs': sum(t['transaction_cost'] for t in self.transaction_history),
            'data_points': len(self.portfolio_values)
        }
        
        if len(self.portfolio_values) > 1:
            metrics = self.get_performance_metrics()
            summary.update(metrics)
            
        return summary
