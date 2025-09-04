"""Base strategy class for portfolio optimization strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class OptimizationStrategy(ABC):
    """Abstract base class for portfolio optimization strategies."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            **kwargs: Strategy-specific parameters
        """
        self.name = name
        self.parameters = kwargs
        
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Optimize portfolio weights.
        
        Args:
            returns: DataFrame with date index and ticker columns
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary mapping ticker to optimal weight
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        pass
    
    def validate_inputs(self, returns: pd.DataFrame) -> None:
        """Validate input data.
        
        Args:
            returns: Returns DataFrame to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")
            
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns DataFrame must have DatetimeIndex")
            
        if len(returns.columns) == 0:
            raise ValueError("Returns DataFrame has no columns")
            
        # Check for excessive missing data
        missing_counts = np.sum(returns.isnull(), axis=0)
        missing_pct = missing_counts / len(returns)
        high_missing = missing_pct[missing_pct > 0.5]
        if not high_missing.empty:
            raise ValueError(f"Too much missing data for tickers: {list(high_missing.index)}")
    
    def preprocess_returns(self, returns: pd.DataFrame, 
                          method: str = "forward_fill") -> pd.DataFrame:
        """Preprocess returns data.
        
        Args:
            returns: Raw returns DataFrame
            method: Method for handling missing values ("forward_fill", "drop", "interpolate")
            
        Returns:
            Preprocessed returns DataFrame
        """
        if method == "forward_fill":
            return returns.ffill().fillna(0)
        elif method == "drop":
            return returns.dropna()
        elif method == "interpolate":
            return returns.interpolate().fillna(0)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def calculate_expected_returns(self, returns: pd.DataFrame, 
                                 method: str = "mean") -> pd.Series:
        """Calculate expected returns.
        
        Args:
            returns: Returns DataFrame
            method: Method for calculating expected returns ("mean", "geometric_mean", "median")
            
        Returns:
            Series of expected returns
        """
        if method == "mean":
            return returns.mean()
        elif method == "geometric_mean":
            return (1 + returns).prod() ** (1 / len(returns)) - 1
        elif method == "median":
            return returns.median()
        else:
            raise ValueError(f"Unknown expected returns method: {method}")
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame, 
                                  method: str = "sample") -> pd.DataFrame:
        """Calculate covariance matrix.
        
        Args:
            returns: Returns DataFrame
            method: Method for calculating covariance ("sample", "exponential", "robust")
            
        Returns:
            Covariance matrix DataFrame
        """
        if method == "sample":
            return returns.cov()
        elif method == "exponential":
            # Exponential weighted covariance
            lambda_param = 0.94  # RiskMetrics lambda
            return returns.ewm(alpha=1-lambda_param).cov()
        elif method == "robust":
            # Robust covariance using minimum covariance determinant
            from sklearn.covariance import MinCovDet
            robust_cov = MinCovDet().fit(returns.dropna())
            return pd.DataFrame(robust_cov.covariance_, 
                              index=returns.columns, 
                              columns=returns.columns)
        else:
            raise ValueError(f"Unknown covariance method: {method}")
    
    def apply_constraints(self, weights: Dict[str, float], 
                         constraints: Dict[str, Any]) -> Dict[str, float]:
        """Apply constraints to weights.
        
        Args:
            weights: Raw weights
            constraints: Dictionary of constraints
            
        Returns:
            Constrained weights
        """
        constrained_weights = weights.copy()
        
        # Apply minimum weight constraints
        min_weight = constraints.get('min_weight', 0.0)
        for ticker in constrained_weights:
            constrained_weights[ticker] = max(constrained_weights[ticker], min_weight)
            
        # Apply maximum weight constraints
        max_weight = constraints.get('max_weight', 1.0)
        for ticker in constrained_weights:
            constrained_weights[ticker] = min(constrained_weights[ticker], max_weight)
            
        # Apply sector constraints if specified
        sector_constraints = constraints.get('sector_constraints', {})
        if sector_constraints:
            # This would require sector mapping - simplified for now
            pass
            
        # Normalize weights to sum to 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for ticker in constrained_weights:
                constrained_weights[ticker] /= total_weight
                
        return constrained_weights
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} Strategy"
    
    def __repr__(self) -> str:
        """Detailed string representation of strategy."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}(name='{self.name}', {params_str})"


class StrategyFactory:
    """Factory for creating optimization strategies."""
    
    _strategies = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a strategy class.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class to register
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> OptimizationStrategy:
        """Create a strategy instance.
        
        Args:
            name: Strategy name
            **kwargs: Strategy parameters
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")
            
        strategy_class = cls._strategies[name]
        return strategy_class(name=name, **kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategies.
        
        Returns:
            List of available strategy names
        """
        return list(cls._strategies.keys())
