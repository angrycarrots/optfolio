"""Random weight portfolio optimization strategy."""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from .base import OptimizationStrategy, StrategyFactory


class RandomWeightStrategy(OptimizationStrategy):
    """Random weight portfolio optimization strategy."""
    
    def __init__(self, name: str = "Random Weight", 
                 seed: Optional[int] = None,
                 distribution: str = "dirichlet",
                 **kwargs):
        """Initialize random weight strategy.
        
        Args:
            name: Strategy name
            seed: Random seed for reproducibility
            distribution: Distribution for generating weights ("dirichlet", "uniform", "normal")
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.seed = seed
        self.distribution = distribution
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Generate random portfolio weights.
        
        Args:
            returns: DataFrame with date index and ticker columns
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping ticker to random weight
        """
        # Validate inputs
        self.validate_inputs(returns)
        
        # Get list of tickers
        tickers = list(returns.columns)
        
        if not tickers:
            raise ValueError("No tickers available for optimization")
        
        # Generate random weights based on distribution
        if self.distribution == "dirichlet":
            weights = self._generate_dirichlet_weights(len(tickers))
        elif self.distribution == "uniform":
            weights = self._generate_uniform_weights(len(tickers))
        elif self.distribution == "normal":
            weights = self._generate_normal_weights(len(tickers))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        # Create weights dictionary
        weight_dict = dict(zip(tickers, weights))
        
        # Apply constraints if specified
        constraints = kwargs.get('constraints', {})
        if constraints:
            weight_dict = self.apply_constraints(weight_dict, constraints)
        
        return weight_dict
    
    def _generate_dirichlet_weights(self, n_assets: int) -> np.ndarray:
        """Generate weights using Dirichlet distribution.
        
        Args:
            n_assets: Number of assets
            
        Returns:
            Array of weights that sum to 1
        """
        # Dirichlet distribution naturally generates weights that sum to 1
        alpha = np.ones(n_assets)  # Uniform Dirichlet distribution
        weights = np.random.dirichlet(alpha)
        return weights
    
    def _generate_uniform_weights(self, n_assets: int) -> np.ndarray:
        """Generate weights using uniform distribution.
        
        Args:
            n_assets: Number of assets
            
        Returns:
            Array of weights that sum to 1
        """
        # Generate uniform random numbers
        weights = np.random.uniform(0, 1, n_assets)
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        return weights
    
    def _generate_normal_weights(self, n_assets: int) -> np.ndarray:
        """Generate weights using normal distribution.
        
        Args:
            n_assets: Number of assets
            
        Returns:
            Array of weights that sum to 1
        """
        # Generate normal random numbers
        weights = np.random.normal(0, 1, n_assets)
        
        # Take absolute values to ensure non-negative weights
        weights = np.abs(weights)
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        return weights
    
    def generate_multiple_weights(self, returns: pd.DataFrame, 
                                n_portfolios: int = 100,
                                **kwargs) -> List[Dict[str, float]]:
        """Generate multiple random portfolios.
        
        Args:
            returns: DataFrame with date index and ticker columns
            n_portfolios: Number of portfolios to generate
            **kwargs: Additional parameters
            
        Returns:
            List of weight dictionaries
        """
        portfolios = []
        
        for i in range(n_portfolios):
            # Set seed for reproducibility if base seed is provided
            if self.seed is not None:
                np.random.seed(self.seed + i)
            
            weights = self.optimize(returns, **kwargs)
            portfolios.append(weights)
        
        return portfolios
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            'strategy_type': 'random_weight',
            'distribution': self.distribution,
            'seed': self.seed,
            'description': f'Random weight allocation using {self.distribution} distribution',
            **self.parameters
        }
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} Strategy ({self.distribution})"


# Register the strategy
StrategyFactory.register("random_weight", RandomWeightStrategy)
