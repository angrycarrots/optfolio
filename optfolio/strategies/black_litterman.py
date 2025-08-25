"""Black-Litterman portfolio optimization strategy."""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from .base import OptimizationStrategy, StrategyFactory


class BlackLittermanStrategy(OptimizationStrategy):
    """Black-Litterman portfolio optimization strategy."""
    
    def __init__(self, name: str = "Black-Litterman", 
                 tau: float = 0.05,
                 risk_aversion: float = 2.5,
                 prior_method: str = "market_cap",
                 view_method: str = "random",
                 **kwargs):
        """Initialize Black-Litterman strategy.
        
        Args:
            name: Strategy name
            tau: Prior uncertainty parameter
            risk_aversion: Risk aversion parameter
            prior_method: Method for generating priors ("market_cap", "equal", "random")
            view_method: Method for generating views ("random", "momentum", "mean_reversion")
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.prior_method = prior_method
        self.view_method = view_method
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Optimize portfolio weights using Black-Litterman model.
        
        Args:
            returns: DataFrame with date index and ticker columns
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary mapping ticker to optimal weight
        """
        # Validate inputs
        self.validate_inputs(returns)
        
        # Preprocess returns
        preprocess_method = kwargs.get('preprocess_method', 'forward_fill')
        processed_returns = self.preprocess_returns(returns, preprocess_method)
        
        # Calculate market equilibrium returns and covariance
        equilibrium_returns = self._calculate_equilibrium_returns(processed_returns)
        covariance = self.calculate_covariance_matrix(processed_returns, 
                                                    method=kwargs.get('covariance_method', 'sample'))
        
        # Generate views
        views, confidence = self._generate_views(processed_returns, **kwargs)
        
        # Apply Black-Litterman model
        posterior_returns, posterior_covariance = self._black_litterman_model(
            equilibrium_returns, covariance, views, confidence
        )
        
        # Optimize using posterior estimates
        weights = self._optimize_with_posterior(posterior_returns, posterior_covariance, **kwargs)
        
        # Apply constraints if specified
        constraints = kwargs.get('constraints', {})
        if constraints:
            weights = self.apply_constraints(weights, constraints)
        
        return weights
    
    def _calculate_equilibrium_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate market equilibrium returns.
        
        Args:
            returns: Returns DataFrame
            
        Returns:
            Series of equilibrium returns
        """
        # Calculate covariance matrix
        covariance = returns.cov()
        
        # Generate market weights based on prior method
        if self.prior_method == "market_cap":
            # Simulate market cap weights (in practice, you'd use actual market caps)
            market_weights = self._generate_market_cap_weights(len(returns.columns))
        elif self.prior_method == "equal":
            market_weights = np.ones(len(returns.columns)) / len(returns.columns)
        elif self.prior_method == "random":
            market_weights = np.random.dirichlet(np.ones(len(returns.columns)))
        else:
            raise ValueError(f"Unknown prior method: {self.prior_method}")
        
        # Calculate equilibrium returns using reverse optimization
        # π = λ * Σ * w_market
        equilibrium_returns = self.risk_aversion * covariance.dot(market_weights)
        
        return pd.Series(equilibrium_returns, index=returns.columns)
    
    def _generate_market_cap_weights(self, n_assets: int) -> np.ndarray:
        """Generate simulated market cap weights.
        
        Args:
            n_assets: Number of assets
            
        Returns:
            Array of market cap weights
        """
        # Simulate market caps using log-normal distribution
        market_caps = np.random.lognormal(mean=10, sigma=1, size=n_assets)
        weights = market_caps / np.sum(market_caps)
        return weights
    
    def _generate_views(self, returns: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate views and confidence levels.
        
        Args:
            returns: Returns DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (views matrix, confidence levels)
        """
        n_assets = len(returns.columns)
        
        if self.view_method == "random":
            return self._generate_random_views(n_assets, **kwargs)
        elif self.view_method == "momentum":
            return self._generate_momentum_views(returns, **kwargs)
        elif self.view_method == "mean_reversion":
            return self._generate_mean_reversion_views(returns, **kwargs)
        else:
            raise ValueError(f"Unknown view method: {self.view_method}")
    
    def _generate_random_views(self, n_assets: int, 
                             n_views: int = 3,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random views.
        
        Args:
            n_assets: Number of assets
            n_views: Number of views to generate
            
        Returns:
            Tuple of (views matrix, confidence levels)
        """
        # Generate random views matrix
        views_matrix = np.zeros((n_views, n_assets))
        
        for i in range(n_views):
            # Randomly select assets for this view
            n_assets_in_view = np.random.randint(1, min(4, n_assets + 1))
            assets_in_view = np.random.choice(n_assets, n_assets_in_view, replace=False)
            
            # Generate random weights for selected assets
            weights = np.random.uniform(-1, 1, n_assets_in_view)
            weights = weights / np.sum(np.abs(weights))  # Normalize
            
            # Set view weights
            views_matrix[i, assets_in_view] = weights
        
        # Generate random confidence levels
        confidence_levels = np.random.uniform(0.1, 0.9, n_views)
        
        return views_matrix, confidence_levels
    
    def _generate_momentum_views(self, returns: pd.DataFrame, 
                               lookback_period: int = 60,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate momentum-based views.
        
        Args:
            returns: Returns DataFrame
            lookback_period: Period for momentum calculation
            
        Returns:
            Tuple of (views matrix, confidence levels)
        """
        # Calculate momentum (recent performance)
        recent_returns = returns.tail(lookback_period).mean()
        
        # Sort assets by momentum
        sorted_assets = recent_returns.sort_values(ascending=False)
        
        # Create views: overweight top performers, underweight bottom performers
        n_assets = len(returns.columns)
        n_views = min(3, n_assets // 2)
        
        views_matrix = np.zeros((n_views, n_assets))
        confidence_levels = np.zeros(n_views)
        
        for i in range(n_views):
            if i < len(sorted_assets) // 2:
                # Positive view on top performer
                top_asset_idx = returns.columns.get_loc(sorted_assets.index[i])
                views_matrix[i, top_asset_idx] = 1.0
                confidence_levels[i] = 0.7
            else:
                # Negative view on bottom performer
                bottom_asset_idx = returns.columns.get_loc(sorted_assets.index[-(i+1)])
                views_matrix[i, bottom_asset_idx] = -1.0
                confidence_levels[i] = 0.6
        
        return views_matrix, confidence_levels
    
    def _generate_mean_reversion_views(self, returns: pd.DataFrame,
                                     lookback_period: int = 252,
                                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mean reversion views.
        
        Args:
            returns: Returns DataFrame
            lookback_period: Period for mean calculation
            
        Returns:
            Tuple of (views matrix, confidence levels)
        """
        # Calculate long-term mean returns
        long_term_means = returns.tail(lookback_period).mean()
        recent_returns = returns.tail(20).mean()  # Recent performance
        
        # Find assets that have deviated most from their long-term mean
        deviations = recent_returns - long_term_means
        sorted_deviations = deviations.sort_values()
        
        n_assets = len(returns.columns)
        n_views = min(3, n_assets // 2)
        
        views_matrix = np.zeros((n_views, n_assets))
        confidence_levels = np.zeros(n_views)
        
        for i in range(n_views):
            if i < len(sorted_deviations) // 2:
                # Positive view on underperforming asset (mean reversion)
                asset_idx = returns.columns.get_loc(sorted_deviations.index[i])
                views_matrix[i, asset_idx] = 1.0
                confidence_levels[i] = 0.5
            else:
                # Negative view on overperforming asset (mean reversion)
                asset_idx = returns.columns.get_loc(sorted_deviations.index[-(i+1)])
                views_matrix[i, asset_idx] = -1.0
                confidence_levels[i] = 0.5
        
        return views_matrix, confidence_levels
    
    def _black_litterman_model(self, equilibrium_returns: pd.Series,
                             covariance: pd.DataFrame,
                             views_matrix: np.ndarray,
                             confidence_levels: np.ndarray) -> Tuple[pd.Series, pd.DataFrame]:
        """Apply Black-Litterman model.
        
        Args:
            equilibrium_returns: Market equilibrium returns
            covariance: Asset covariance matrix
            views_matrix: Views matrix
            confidence_levels: Confidence levels for views
            
        Returns:
            Tuple of (posterior returns, posterior covariance)
        """
        n_assets = len(equilibrium_returns)
        n_views = len(confidence_levels)
        
        # Prior covariance
        prior_covariance = self.tau * covariance
        
        # Views uncertainty matrix
        omega = np.diag(1 / confidence_levels)
        
        # Posterior covariance
        # Σ_post = ((τΣ)^(-1) + P^T Ω^(-1) P)^(-1)
        prior_inv = np.linalg.inv(prior_covariance)
        omega_inv = np.linalg.inv(omega)
        
        posterior_covariance = np.linalg.inv(
            prior_inv + views_matrix.T @ omega_inv @ views_matrix
        )
        
        # Posterior returns
        # μ_post = Σ_post * ((τΣ)^(-1) * π + P^T * Ω^(-1) * Q)
        # For simplicity, we'll use equilibrium returns as views
        view_returns = equilibrium_returns.values  # In practice, these would be actual views
        
        # Ensure view_returns has the right shape for matrix multiplication
        if len(view_returns) != n_views:
            # If view_returns doesn't match n_views, use equilibrium returns for each view
            view_returns = np.array([equilibrium_returns.values.mean()] * n_views)
        
        posterior_returns = posterior_covariance @ (
            prior_inv @ equilibrium_returns.values + 
            views_matrix.T @ omega_inv @ view_returns
        )
        
        return pd.Series(posterior_returns, index=equilibrium_returns.index), pd.DataFrame(posterior_covariance)
    
    def _optimize_with_posterior(self, posterior_returns: pd.Series,
                               posterior_covariance: pd.DataFrame,
                               **kwargs) -> Dict[str, float]:
        """Optimize weights using posterior estimates.
        
        Args:
            posterior_returns: Posterior expected returns
            posterior_covariance: Posterior covariance matrix
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of optimal weights
        """
        from scipy.optimize import minimize
        
        tickers = list(posterior_returns.index)
        n_assets = len(tickers)
        
        # Risk-free rate (daily)
        rf_daily = kwargs.get('risk_free_rate', 0.02) / 252
        
        def objective(weights):
            """Maximize Sharpe ratio."""
            weights = np.array(weights)
            
            # Portfolio return
            portfolio_return = np.sum(weights * posterior_returns)
            
            # Portfolio volatility
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(posterior_covariance, weights)))
            
            # Sharpe ratio
            if portfolio_volatility > 0:
                return -(portfolio_return - rf_daily) / portfolio_volatility
            else:
                return -np.inf
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, 
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(tickers, result.x))
        else:
            # Fallback to equal weights
            weights = {ticker: 1.0/n_assets for ticker in tickers}
        
        return weights
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            'strategy_type': 'black_litterman',
            'tau': self.tau,
            'risk_aversion': self.risk_aversion,
            'prior_method': self.prior_method,
            'view_method': self.view_method,
            'description': f'Black-Litterman model with {self.prior_method} priors and {self.view_method} views',
            **self.parameters
        }
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} Strategy ({self.prior_method} priors, {self.view_method} views)"


# Register the strategy
StrategyFactory.register("black_litterman", BlackLittermanStrategy)
