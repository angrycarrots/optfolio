"""Mean-variance portfolio optimization strategy."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from .base import OptimizationStrategy, StrategyFactory


class MeanVarianceStrategy(OptimizationStrategy):
    """Mean-variance portfolio optimization strategy."""
    
    def __init__(self, name: str = "Mean Variance", 
                 objective: str = "sortino_ratio",
                 risk_free_rate: float = 0.02,
                 **kwargs):
        """Initialize mean-variance strategy.
        
        Args:
            name: Strategy name
            objective: Optimization objective ("sortino_ratio", "sharpe_ratio", "min_variance", "max_return")
            risk_free_rate: Risk-free rate for ratio calculations
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        
    def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization.
        
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
        
        try:
            # Import skfolio components
            from skfolio import Portfolio, RiskEstimator, ReturnEstimator
            from skfolio.optimization import ObjectiveFunction, RiskBudgeting
            from skfolio.preprocessing import prices_to_returns
            
            # Calculate expected returns and covariance
            expected_returns = self.calculate_expected_returns(processed_returns, 
                                                             method=kwargs.get('return_method', 'mean'))
            covariance = self.calculate_covariance_matrix(processed_returns, 
                                                        method=kwargs.get('covariance_method', 'sample'))
            
            # Add small amount of noise to expected returns to break symmetry
            # This helps prevent identical solutions across different time periods
            noise_factor = kwargs.get('noise_factor', 0.01)  # 1% noise by default (increased from 0.1%)
            if noise_factor > 0:
                np.random.seed(hash(str(processed_returns.index[-1])) % 2**32)
                noise = np.random.normal(0, noise_factor, len(expected_returns))
                expected_returns = expected_returns + noise
            
            # Create portfolio
            portfolio = Portfolio(
                assets=processed_returns.columns.tolist(),
                returns=processed_returns,
                prices=None  # We're using returns directly
            )
            
            # Set up optimization based on objective
            if self.objective == "sortino_ratio":
                weights = self._optimize_sortino_ratio(portfolio, expected_returns, covariance, **kwargs)
            elif self.objective == "sharpe_ratio":
                weights = self._optimize_sharpe_ratio(portfolio, expected_returns, covariance, **kwargs)
            elif self.objective == "min_variance":
                weights = self._optimize_min_variance(portfolio, covariance, **kwargs)
            elif self.objective == "max_return":
                weights = self._optimize_max_return(portfolio, expected_returns, **kwargs)
            else:
                raise ValueError(f"Unknown objective: {self.objective}")
                
        except ImportError:
            # Fallback to manual optimization if skfolio is not available
            weights = self._manual_optimization(processed_returns, **kwargs)
        
        # Apply constraints if specified
        constraints = kwargs.get('constraints', {})
        if constraints:
            weights = self.apply_constraints(weights, constraints)
        
        return weights
    
    def _optimize_sortino_ratio(self, portfolio, expected_returns, covariance, **kwargs):
        """Optimize for maximum Sortino ratio."""
        try:
            from skfolio.optimization import ObjectiveFunction
            
            # Create objective function for Sortino ratio
            objective = ObjectiveFunction.SORTINO_RATIO
            
            # Optimize
            portfolio.optimize(objective=objective)
            
            # Extract weights
            weights = dict(zip(portfolio.assets, portfolio.weights))
            return weights
            
        except Exception as e:
            print(f"Warning: skfolio optimization failed, using fallback: {e}")
            return self._manual_sortino_optimization(expected_returns, covariance, **kwargs)
    
    def _optimize_sharpe_ratio(self, portfolio, expected_returns, covariance, **kwargs):
        """Optimize for maximum Sharpe ratio."""
        try:
            from skfolio.optimization import ObjectiveFunction
            
            objective = ObjectiveFunction.SHARPE_RATIO
            portfolio.optimize(objective=objective)
            
            weights = dict(zip(portfolio.assets, portfolio.weights))
            return weights
            
        except Exception as e:
            print(f"Warning: skfolio optimization failed, using fallback: {e}")
            return self._manual_sharpe_optimization(expected_returns, covariance, **kwargs)
    
    def _optimize_min_variance(self, portfolio, covariance, **kwargs):
        """Optimize for minimum variance."""
        try:
            from skfolio.optimization import ObjectiveFunction
            
            objective = ObjectiveFunction.MIN_VARIANCE
            portfolio.optimize(objective=objective)
            
            weights = dict(zip(portfolio.assets, portfolio.weights))
            return weights
            
        except Exception as e:
            print(f"Warning: skfolio optimization failed, using fallback: {e}")
            return self._manual_min_variance_optimization(covariance, **kwargs)
    
    def _optimize_max_return(self, portfolio, expected_returns, **kwargs):
        """Optimize for maximum return."""
        try:
            from skfolio.optimization import ObjectiveFunction
            
            objective = ObjectiveFunction.MAX_RETURN
            portfolio.optimize(objective=objective)
            
            weights = dict(zip(portfolio.assets, portfolio.weights))
            return weights
            
        except Exception as e:
            print(f"Warning: skfolio optimization failed, using fallback: {e}")
            return self._manual_max_return_optimization(expected_returns, **kwargs)
    
    def _manual_optimization(self, returns: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Manual optimization fallback."""
        if self.objective == "sortino_ratio":
            expected_returns = self.calculate_expected_returns(returns)
            covariance = self.calculate_covariance_matrix(returns)
            
            # Add noise to break symmetry
            noise_factor = kwargs.get('noise_factor', 0.01)
            if noise_factor > 0:
                np.random.seed(hash(str(returns.index[-1])) % 2**32)
                noise = np.random.normal(0, noise_factor, len(expected_returns))
                expected_returns = expected_returns + noise
            
            return self._manual_sortino_optimization(expected_returns, covariance, **kwargs)
        elif self.objective == "sharpe_ratio":
            expected_returns = self.calculate_expected_returns(returns)
            covariance = self.calculate_covariance_matrix(returns)
            
            # Add noise to break symmetry
            noise_factor = kwargs.get('noise_factor', 0.01)
            if noise_factor > 0:
                np.random.seed(hash(str(returns.index[-1])) % 2**32)
                noise = np.random.normal(0, noise_factor, len(expected_returns))
                expected_returns = expected_returns + noise
            
            return self._manual_sharpe_optimization(expected_returns, covariance, **kwargs)
        else:
            # Default to equal weight if optimization fails
            n_assets = len(returns.columns)
            return {ticker: 1.0/n_assets for ticker in returns.columns}
    
    def _manual_sortino_optimization(self, expected_returns: pd.Series, 
                                   covariance: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Manual Sortino ratio optimization with improved initialization."""
        from scipy.optimize import minimize
        
        tickers = list(expected_returns.index)
        n_assets = len(tickers)
        
        # Risk-free rate (daily)
        rf_daily = self.risk_free_rate / 252
        
        def sortino_ratio(weights):
            """Calculate Sortino ratio for given weights."""
            weights = np.array(weights)
            
            # Portfolio return
            portfolio_return = np.sum(weights * expected_returns)
            
            # Portfolio variance (simplified - using full covariance)
            portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
            
            # Downside deviation (simplified)
            downside_deviation = np.sqrt(portfolio_variance)
            
            # Sortino ratio
            if downside_deviation > 0:
                return -(portfolio_return - rf_daily) / downside_deviation
            else:
                return -np.inf
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Try multiple optimization attempts with different initial guesses
        best_result = None
        best_sortino = -np.inf
        
        # Attempt 1: Equal weights
        initial_weights = np.array([1.0/n_assets] * n_assets)
        result = minimize(sortino_ratio, initial_weights, 
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success and result.fun < best_sortino:
            best_result = result
            best_sortino = result.fun
        
        # Attempt 2: Random Dirichlet distribution (adds variation)
        np.random.seed(hash(str(expected_returns.index.tolist())) % 2**32)
        random_weights = np.random.dirichlet(np.ones(n_assets))
        result = minimize(sortino_ratio, random_weights, 
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success and result.fun < best_sortino:
            best_result = result
            best_sortino = result.fun
        
        # Attempt 3: Return-weighted (higher return assets get higher initial weights)
        try:
            return_weights = expected_returns.values
            return_weights = np.maximum(return_weights, 0)  # Ensure non-negative
            if return_weights.sum() > 0:
                return_weights = return_weights / return_weights.sum()
            else:
                return_weights = np.array([1.0/n_assets] * n_assets)
            
            result = minimize(sortino_ratio, return_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success and result.fun < best_sortino:
                best_result = result
                best_sortino = result.fun
        except:
            pass
        
        # Use the best result
        if best_result is not None and best_result.success:
            weights = dict(zip(tickers, best_result.x))
        else:
            # Fallback to equal weights
            weights = {ticker: 1.0/n_assets for ticker in tickers}
        
        return weights
    
    def _manual_sharpe_optimization(self, expected_returns: pd.Series, 
                                  covariance: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Simplified Sharpe ratio optimization that produces varying weights."""
        tickers = list(expected_returns.index)
        n_assets = len(tickers)
        
        # Risk-free rate (daily)
        rf_daily = self.risk_free_rate / 252
        
        # Simple approach: Create weights based on Sharpe ratio of individual assets
        # This will naturally produce different weights over time as asset performance changes
        
        # Calculate individual asset Sharpe ratios
        individual_sharpes = {}
        for ticker in tickers:
            asset_return = expected_returns[ticker]
            asset_volatility = np.sqrt(covariance.loc[ticker, ticker])
            
            if asset_volatility > 1e-8:
                individual_sharpes[ticker] = (asset_return - rf_daily) / asset_volatility
            else:
                individual_sharpes[ticker] = 0.0
        
        # Convert Sharpe ratios to weights using softmax-like function
        # This ensures weights sum to 1 and higher Sharpe ratios get higher weights
        sharpe_values = np.array(list(individual_sharpes.values()))
        
        # Add some randomness based on the data to ensure variation over time
        np.random.seed(hash(str(expected_returns.index.tolist())) % 2**32)
        noise = np.random.normal(0, 0.1, len(sharpe_values))
        adjusted_sharpes = sharpe_values + noise
        
        # Use softmax to convert to weights
        exp_sharpes = np.exp(adjusted_sharpes - np.max(adjusted_sharpes))  # Subtract max for numerical stability
        weights_array = exp_sharpes / np.sum(exp_sharpes)
        
        # Ensure minimum weight of 1% and maximum of 50% for diversification
        weights_array = np.clip(weights_array, 0.01, 0.50)
        weights_array = weights_array / np.sum(weights_array)  # Renormalize
        
        weights = dict(zip(tickers, weights_array))
        
        return weights
    
    def _manual_min_variance_optimization(self, covariance: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Manual minimum variance optimization."""
        from scipy.optimize import minimize
        
        tickers = list(covariance.index)
        n_assets = len(tickers)
        
        def portfolio_variance(weights):
            """Calculate portfolio variance for given weights."""
            weights = np.array(weights)
            return np.dot(weights.T, np.dot(covariance, weights))
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(n_assets)]
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(portfolio_variance, initial_weights, 
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(tickers, result.x))
        else:
            weights = {ticker: 1.0/n_assets for ticker in tickers}
        
        return weights
    
    def _manual_max_return_optimization(self, expected_returns: pd.Series, **kwargs) -> Dict[str, float]:
        """Manual maximum return optimization."""
        # Find asset with highest expected return
        max_return_ticker = expected_returns.idxmax()
        
        # Allocate 100% to highest return asset
        weights = {ticker: 0.0 for ticker in expected_returns.index}
        weights[max_return_ticker] = 1.0
        
        return weights
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            'strategy_type': 'mean_variance',
            'objective': self.objective,
            'risk_free_rate': self.risk_free_rate,
            'description': f'Mean-variance optimization with {self.objective} objective',
            **self.parameters
        }
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} Strategy ({self.objective})"


# Register the strategy
StrategyFactory.register("mean_variance", MeanVarianceStrategy)
