"""Black-Litterman portfolio optimization strategy."""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from .base import OptimizationStrategy, StrategyFactory
from .upside import UpsideCalculator


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
            view_method: Method for generating views ("random", "momentum", "mean_reversion", "upside")
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.prior_method = prior_method
        self.view_method = view_method
        
        # Initialize rebalancing log for tracking weights and upside data
        self.rebalancing_log = []
    
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
        
        # Ensure covariance matrix is positive definite and finite
        covariance = covariance + np.eye(len(covariance)) * 1e-8
        covariance = pd.DataFrame(covariance, index=covariance.index, columns=covariance.columns)
        
        # Ensure equilibrium returns are finite
        equilibrium_returns = equilibrium_returns.replace([np.inf, -np.inf], 0.0)
        equilibrium_returns = equilibrium_returns.fillna(0.0)
        
        # Generate views
        views, confidence = self._generate_views(processed_returns, **kwargs)
        
        # Ensure views and confidence are finite
        views = np.nan_to_num(views, nan=0.0, posinf=1.0, neginf=-1.0)
        confidence = np.nan_to_num(confidence, nan=0.5, posinf=0.9, neginf=0.1)
        
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
        
        # Log rebalancing data if this is an upside strategy
        if self.view_method == "upside":
            self._log_rebalancing_data(returns, weights, **kwargs)
        
        return weights
    
    def _log_rebalancing_data(self, returns: pd.DataFrame, weights: Dict[str, float], **kwargs):
        """Log rebalancing data including weights and upside values for CSV export.
        
        Args:
            returns: Returns DataFrame used for optimization
            weights: Optimized portfolio weights
            **kwargs: Additional parameters
        """
        if self.view_method != "upside":
            return
            
        try:
            # Get current date (use the last date in returns data)
            current_date = returns.index[-1] if not returns.empty else pd.Timestamp.now()
            
            # Get upside data for all symbols
            uc = UpsideCalculator()
            upside_data = {}
            
            for symbol in returns.columns:
                try:
                    symbol_upside = uc.upside(symbol)
                    if not symbol_upside.empty:
                        # Get upside value at the specific rebalancing date
                        # Find the closest date that's <= current_date
                        available_dates = symbol_upside.index
                        valid_dates = available_dates[available_dates <= current_date]
                        
                        if len(valid_dates) > 0:
                            # Get the most recent available date <= current_date
                            target_date = valid_dates[-1]
                            date_upside = symbol_upside.loc[target_date]['upside']
                            
                            # Handle case where multiple entries exist for the same date
                            if isinstance(date_upside, pd.Series):
                                # Take the last (most recent) value if multiple entries
                                date_upside = date_upside.iloc[-1]
                            
                            # Ensure we have a single numeric value
                            if pd.isna(date_upside) or not np.isfinite(date_upside):
                                upside_data[symbol] = np.nan
                            else:
                                upside_data[symbol] = float(date_upside)
                        else:
                            # No data available at or before current_date
                            upside_data[symbol] = np.nan
                    else:
                        upside_data[symbol] = np.nan
                except Exception as e:
                    upside_data[symbol] = np.nan
            
            # Create log entry
            log_entry = {
                'date': current_date,
                'weights': weights.copy(),
                'upside_values': upside_data.copy()
            }
            
            self.rebalancing_log.append(log_entry)
            
        except Exception as e:
            print(f"Warning: Could not log rebalancing data: {e}")
    
    def get_rebalancing_log(self) -> List[Dict]:
        """Get the rebalancing log for CSV export.
        
        Returns:
            List of rebalancing log entries
        """
        return self.rebalancing_log
    
    def export_rebalancing_data_to_csv(self, filename: str = None) -> str:
        """Export rebalancing data to CSV file.
        
        Args:
            filename: Output filename (default: auto-generated)
            
        Returns:
            Filename of the exported CSV file
        """
        if not self.rebalancing_log:
            print("Warning: No rebalancing data to export")
            return None
            
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"black_litterman_upside_rebalancing_{timestamp}.csv"
        
        try:
            # Prepare data for CSV export
            csv_data = []
            
            for log_entry in self.rebalancing_log:
                date = log_entry['date']
                weights = log_entry['weights']
                upside_values = log_entry['upside_values']
                
                # Create row for each symbol
                for symbol in weights.keys():
                    row = {
                        'Date': date,
                        'Symbol': symbol,
                        'Weight': weights[symbol],
                        'Upside': upside_values.get(symbol, np.nan)
                    }
                    csv_data.append(row)
            
            # Create DataFrame and export
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
            
            print(f"Exported rebalancing data to: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting rebalancing data: {e}")
            return None
    
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
        elif self.view_method == "upside":
            return self._generate_upside_views(returns, **kwargs)
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
    
    def _generate_upside_views(self, returns: pd.DataFrame, 
                              symbols: List[str] = None,
                              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate views based on analyst price target upside.
        
        Args:
            returns: Returns DataFrame
            symbols: List of symbols to generate views for (default: all)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (views matrix, confidence levels)
        """
        if symbols is None:
            symbols = list(returns.columns)
        
        # Initialize UpsideCalculator
        uc = UpsideCalculator()
        
        n_assets = len(returns.columns)
        n_views = len(symbols)  # One view per symbol
        
        views_matrix = np.zeros((n_views, n_assets))
        confidence_levels = np.zeros(n_views)
        
        successful_views = 0
        
        for i, symbol in enumerate(symbols):
            try:
                # Get upside data for this symbol
                upside_data = uc.upside(symbol)
                
                if not upside_data.empty:
                    # Get the most recent upside
                    latest_upside = upside_data.iloc[-1]['upside']
                    
                    # Find the column index for this symbol
                    if symbol in returns.columns:
                        col_idx = returns.columns.get_loc(symbol)
                        
                        # Set the view: positive if upside > 0, negative if < 0
                        views_matrix[i, col_idx] = 1.0 if latest_upside > 0 else -1.0
                        
                        # Set confidence based on upside magnitude (higher upside = higher confidence)
                        confidence_levels[i] = min(0.9, max(0.1, abs(latest_upside)))
                        successful_views += 1
                    else:
                        # Symbol not in returns, skip this view
                        views_matrix[i, :] = 0
                        confidence_levels[i] = 0.1
                else:
                    # No upside data, set low confidence
                    views_matrix[i, :] = 0
                    confidence_levels[i] = 0.1
                    
            except Exception as e:
                print(f"Warning: Could not generate upside view for {symbol}: {e}")
                views_matrix[i, :] = 0
                confidence_levels[i] = 0.1
        
        # If no upside views could be generated, fall back to momentum views
        if successful_views == 0:
            print("Warning: No upside views could be generated. Falling back to momentum views.")
            return self._generate_momentum_views(returns, **kwargs)
        
        print(f"Successfully generated {successful_views} upside views out of {len(symbols)} symbols")
        
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
        
        # Add small regularization to ensure positive definiteness
        prior_covariance = prior_covariance + np.eye(n_assets) * 1e-8
        
        # Views uncertainty matrix - ensure no division by zero
        confidence_levels = np.maximum(confidence_levels, 1e-6)  # Minimum confidence level
        omega = np.diag(1 / confidence_levels)
        
        try:
            # Posterior covariance
            # Σ_post = ((τΣ)^(-1) + P^T Ω^(-1) P)^(-1)
            
            # Use more robust matrix inversion with condition number checking
            def safe_inv(matrix, name="matrix"):
                """Safely invert a matrix with condition number checking."""
                try:
                    # Check condition number
                    cond = np.linalg.cond(matrix)
                    if cond > 1e12:  # Very ill-conditioned
                        print(f"Warning: {name} is ill-conditioned (condition number: {cond:.2e})")
                        # Add more regularization
                        matrix = matrix + np.eye(matrix.shape[0]) * 1e-6
                    
                    return np.linalg.inv(matrix)
                except np.linalg.LinAlgError:
                    print(f"Warning: Could not invert {name}, using regularization")
                    # Add regularization and try again
                    matrix = matrix + np.eye(matrix.shape[0]) * 1e-6
                    return np.linalg.inv(matrix)
            
            # Suppress numpy warnings for this section
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                
                prior_inv = safe_inv(prior_covariance, "prior covariance")
                omega_inv = safe_inv(omega, "omega")
                
                # Add regularization to ensure numerical stability
                regularization = np.eye(prior_inv.shape[0]) * 1e-8
                prior_inv = prior_inv + regularization
                
                # Compute the matrix to be inverted with error handling
                try:
                    matrix_to_invert = prior_inv + views_matrix.T @ omega_inv @ views_matrix
                except (ValueError, RuntimeError) as e:
                    print(f"Warning: Matrix multiplication failed: {e}")
                    # Fallback: use only prior information
                    matrix_to_invert = prior_inv
                
                # Add regularization to ensure invertibility
                matrix_to_invert = matrix_to_invert + np.eye(matrix_to_invert.shape[0]) * 1e-8
                
                posterior_covariance = safe_inv(matrix_to_invert, "posterior matrix")
            
            # Posterior returns
            # μ_post = Σ_post * ((τΣ)^(-1) * π + P^T * Ω^(-1) * Q)
            # For simplicity, we'll use equilibrium returns as views
            view_returns = equilibrium_returns.values  # In practice, these would be actual views
            
            # Ensure view_returns has the right shape for matrix multiplication
            if len(view_returns) != n_views:
                # If view_returns doesn't match n_views, use equilibrium returns for each view
                view_returns = np.array([equilibrium_returns.values.mean()] * n_views)
            
            # Ensure view_returns is finite
            view_returns = np.nan_to_num(view_returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute posterior returns with error handling
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                
                try:
                    term1 = prior_inv @ equilibrium_returns.values
                except (ValueError, RuntimeError) as e:
                    print(f"Warning: Term1 calculation failed: {e}")
                    term1 = np.zeros(len(equilibrium_returns))
                
                try:
                    term2 = views_matrix.T @ omega_inv @ view_returns
                except (ValueError, RuntimeError) as e:
                    print(f"Warning: Term2 calculation failed: {e}")
                    term2 = np.zeros(len(equilibrium_returns))
                
                # Ensure terms are finite
                term1 = np.nan_to_num(term1, nan=0.0, posinf=0.0, neginf=0.0)
                term2 = np.nan_to_num(term2, nan=0.0, posinf=0.0, neginf=0.0)
                
                try:
                    posterior_returns = posterior_covariance @ (term1 + term2)
                except (ValueError, RuntimeError) as e:
                    print(f"Warning: Posterior returns calculation failed: {e}")
                    # Fallback to equilibrium returns
                    posterior_returns = equilibrium_returns.values
                
                # Ensure posterior returns are finite
                posterior_returns = np.nan_to_num(posterior_returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            return pd.Series(posterior_returns, index=equilibrium_returns.index), pd.DataFrame(posterior_covariance)
            
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            # Fallback to equilibrium returns and original covariance
            print(f"Warning: Black-Litterman optimization failed, using equilibrium returns. Error: {e}")
            return equilibrium_returns, covariance
    
    def _optimize_with_posterior(self, posterior_returns: pd.Series,
                               posterior_covariance: pd.DataFrame,
                               **kwargs) -> Dict[str, float]:
        """Optimize weights using posterior estimates.
        
        Args:
            posterior_returns: Posterior expected returns
            posterior_covariance: Posterior covariance matrix
            **kwargs: Additional parameters
                - min_weight: Minimum weight per asset (default: 0.01)
                - risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary of optimal weights with minimum weight constraint enforced
        """
        from scipy.optimize import minimize
        
        tickers = list(posterior_returns.index)
        n_assets = len(tickers)
        
        # Risk-free rate (daily)
        rf_daily = kwargs.get('risk_free_rate', 0.02) / 252
        
        def objective(weights):
            """Maximize Sharpe ratio."""
            weights = np.array(weights)
            
            # Ensure weights are finite
            weights = np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Portfolio return
            portfolio_return = np.sum(weights * posterior_returns)
            
            # Portfolio volatility
            try:
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(posterior_covariance, weights)))
                # Ensure volatility is finite and positive
                portfolio_volatility = max(portfolio_volatility, 1e-8)
            except (ValueError, RuntimeError):
                portfolio_volatility = 1e-8
            
            # Sharpe ratio
            if portfolio_volatility > 1e-8:
                sharpe = (portfolio_return - rf_daily) / portfolio_volatility
                # Ensure Sharpe ratio is finite
                if np.isfinite(sharpe):
                    return -sharpe
                else:
                    return -1e6  # Large negative value for invalid Sharpe
            else:
                return -1e6  # Large negative value for zero volatility
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0.01 and 1 (minimum 1% allocation)
        min_weight = kwargs.get('min_weight', 0.01)
        bounds = [(min_weight, 1) for _ in range(n_assets)]
        
        # Initial guess: equal weights (ensuring minimum weight constraint)
        initial_weights = np.array([max(1.0/n_assets, min_weight) for _ in range(n_assets)])
        # Normalize to sum to 1
        initial_weights = initial_weights / np.sum(initial_weights)
        
        # Optimize
        result = minimize(objective, initial_weights, 
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(tickers, result.x))
        else:
            # Fallback to equal weights with minimum constraint
            weights = {ticker: max(1.0/n_assets, min_weight) for ticker in tickers}
            # Normalize to sum to 1
            total_weight = sum(weights.values())
            weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
        # Post-process to ensure minimum weight constraint is met
        weights = self._enforce_minimum_weights(weights, min_weight)
        
        return weights
    
    def _enforce_minimum_weights(self, weights: Dict[str, float], min_weight: float = 0.01) -> Dict[str, float]:
        """Enforce minimum weight constraint and renormalize weights.
        
        Args:
            weights: Dictionary of asset weights
            min_weight: Minimum weight per asset (default: 0.01)
            
        Returns:
            Dictionary of adjusted weights that meet minimum constraint
        """
        weights = weights.copy()
        
        # First pass: ensure all weights meet minimum
        for ticker in weights:
            if weights[ticker] < min_weight:
                weights[ticker] = min_weight
        
        # Calculate total weight
        total_weight = sum(weights.values())
        
        # If total exceeds 1, we need to reduce some weights
        if total_weight > 1.0:
            # Calculate excess weight
            excess = total_weight - 1.0
            
            # Find weights that can be reduced (those above minimum)
            reducible_weights = {ticker: weight for ticker, weight in weights.items() 
                               if weight > min_weight}
            
            if reducible_weights:
                # Sort by weight (reduce largest weights first)
                sorted_reducible = sorted(reducible_weights.items(), key=lambda x: x[1], reverse=True)
                
                # Reduce weights proportionally to their excess above minimum
                for ticker, weight in sorted_reducible:
                    if excess <= 0:
                        break
                    
                    # Calculate how much we can reduce this weight
                    reducible_amount = weight - min_weight
                    reduction = min(excess, reducible_amount)
                    
                    weights[ticker] -= reduction
                    excess -= reduction
        
        # Final normalization to ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
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
