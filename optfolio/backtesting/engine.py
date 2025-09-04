"""Backtesting engine for portfolio optimization strategies."""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import vectorbt as vbt
from scipy import stats

from ..portfolio.base import Portfolio
from ..strategies.base import OptimizationStrategy, StrategyFactory
from ..data.loader import DataLoader
from ..portfolio.metrics import PortfolioMetrics


class Backtester:
    """Main backtesting engine for portfolio strategies."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.02,
                 transaction_costs: float = 0.001):
        """Initialize backtester.
        
        Args:
            initial_capital: Initial portfolio value
            risk_free_rate: Annual risk-free rate
            transaction_costs: Fixed transaction cost per trade (in dollars)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs
        
        # Data storage
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        
        # Results storage
        self.results: Dict[str, Dict] = {}
        
    def load_data(self, data_loader: DataLoader, tickers: Optional[List[str]] = None) -> None:
        """Load price data for backtesting.
        
        Args:
            data_loader: DataLoader instance
            tickers: List of tickers to load (if None, loads all available)
        """
        # Load price data
        self.prices = data_loader.load_prices(tickers)
        
        # Calculate returns
        self.returns = data_loader.get_returns(method="log")
        
        print(f"Loaded data for {len(self.prices.columns)} tickers")
        print(f"Date range: {self.prices.index.min()} to {self.prices.index.max()}")
        print(f"Total observations: {len(self.prices)}")
    
    def run_backtest(self, 
                    strategy: OptimizationStrategy,
                    rebalance_freq: Dict[str, int] = None,
                    start_date: Optional[Union[str, datetime]] = None,
                    end_date: Optional[Union[str, datetime]] = None,
                    **kwargs) -> Dict[str, Any]:
        """Run backtest for a single strategy.
        
        Args:
            strategy: Optimization strategy to test
            rebalance_freq: Rebalancing frequency {"months": N, "weeks": N, "days": N}
            start_date: Start date for backtest
            end_date: End date for backtest
            **kwargs: Additional strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        if self.prices is None or self.returns is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Handle buy-and-hold case (no rebalancing)
        if rebalance_freq is None:
            # For buy-and-hold, we only rebalance once at the beginning
            rebalance_freq = {"months": 0, "weeks": 0, "days": 0}
        
        # Filter data by date range
        prices_filtered = self._filter_by_date(self.prices, start_date, end_date)
        returns_filtered = self._filter_by_date(self.returns, start_date, end_date)
        
        # Create portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
            transaction_costs=self.transaction_costs
        )
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(
            prices_filtered.index, rebalance_freq
        )
        
        print(f"Running backtest for {strategy.name}")
        print(f"Rebalancing dates: {len(rebalance_dates)}")
        print(f"Date range: {prices_filtered.index.min()} to {prices_filtered.index.max()}")
        
        # Run backtest
        results = self._run_backtest_loop(
            portfolio, strategy, prices_filtered, returns_filtered, 
            rebalance_dates, **kwargs
        )
        
        # Store results
        strategy_name = strategy.name
        self.results[strategy_name] = results
        
        return results
    
    def run_multiple_backtests(self, 
                             strategies: List[OptimizationStrategy],
                             rebalance_freq: Dict[str, int] = None,
                             start_date: Optional[Union[str, datetime]] = None,
                             end_date: Optional[Union[str, datetime]] = None,
                             **kwargs) -> Dict[str, Dict]:
        """Run backtests for multiple strategies.
        
        Args:
            strategies: List of optimization strategies to test
            rebalance_freq: Rebalancing frequency
            start_date: Start date for backtest
            end_date: End date for backtest
            **kwargs: Additional strategy parameters
            
        Returns:
            Dictionary with results for all strategies
        """
        all_results = {}
        
        for strategy in strategies:
            try:
                results = self.run_backtest(
                    strategy, rebalance_freq, start_date, end_date, **kwargs
                )
                all_results[strategy.name] = results
            except Exception as e:
                print(f"Error running backtest for {strategy.name}: {e}")
                continue
        
        return all_results
    
    def _filter_by_date(self, data: pd.DataFrame, 
                       start_date: Optional[Union[str, datetime]] = None,
                       end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Filter data by date range."""
        filtered_data = data.copy()
        
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data.index >= start_date]
        
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data.index <= end_date]
        
        return filtered_data
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, 
                           rebalance_freq: Dict[str, int]) -> List[datetime]:
        """Get rebalancing dates based on frequency."""
        months = rebalance_freq.get("months", 1)
        weeks = rebalance_freq.get("weeks", 0)
        days = rebalance_freq.get("days", 0)
        
        # Handle buy-and-hold case (no rebalancing after initial)
        if months == 0 and weeks == 0 and days == 0:
            # For buy-and-hold, use a date that allows for some historical data
            # Use the 30th date to ensure we have enough historical data for optimization
            if len(dates) > 30:
                return [dates[30]]  # Use 30th date to have historical data
            else:
                return [dates[0]]  # Fallback to first date if not enough data
        
        # Convert to pandas frequency string
        if months > 0:
            freq = f"{months}ME"  # Month End
        elif weeks > 0:
            freq = f"{weeks}W"
        elif days > 0:
            freq = f"{days}D"
        else:
            freq = "1ME"  # Default to monthly end
        
        # Generate rebalancing dates using pandas date_range
        rebalance_dates = pd.date_range(
            start=dates[0],
            end=dates[-1],
            freq=freq
        )
        
        # Filter to only include dates that exist in the data
        rebalance_dates = [d for d in rebalance_dates if d in dates]
        
        return rebalance_dates
    
    def _run_backtest_loop(self, portfolio: Portfolio,
                          strategy: OptimizationStrategy,
                          prices: pd.DataFrame,
                          returns: pd.DataFrame,
                          rebalance_dates: List[datetime],
                          **kwargs) -> Dict[str, Any]:
        """Run the main backtest loop."""
        # Initialize portfolio with first rebalance
        if rebalance_dates:
            first_date = rebalance_dates[0]
            
            # Get historical data for initial optimization
            historical_returns = returns[returns.index < first_date]
            
            if len(historical_returns) > 0:
                # Get historical prices for the same period
                historical_prices = prices[prices.index < first_date]
                # Optimize initial weights
                initial_weights = strategy.optimize(historical_returns, prices=historical_prices, **kwargs)
                
                # Get current prices
                current_prices = prices.loc[first_date].to_dict()
                
                # Set initial weights
                portfolio.set_weights(initial_weights)
                
                # Initial rebalance
                portfolio.rebalance(initial_weights, current_prices)
        
        # Step through every trading day for daily portfolio values
        all_dates = prices.index
        
        # Track rebalancing dates for optimization
        rebalance_index = 0
        
        for date in all_dates:
            # Get current prices
            current_prices = prices.loc[date].to_dict()
            
            # Step portfolio forward (this records daily portfolio values)
            portfolio.step(date, current_prices)
            
            # Check if this is a rebalancing date
            if rebalance_dates and rebalance_index < len(rebalance_dates) and date == rebalance_dates[rebalance_index]:
                # This is a rebalancing date - optimize and rebalance
                # For buy-and-hold (single rebalance date), always rebalance
                # For multiple dates, don't rebalance on the last date
                if len(rebalance_dates) == 1 or rebalance_index < len(rebalance_dates) - 1:
                    # Get historical data for optimization (use data up to current date)
                    historical_returns = returns[returns.index <= date]
                    
                    if len(historical_returns) > 30:  # Need at least 30 days of data
                        # Get historical prices for the same period
                        historical_prices = prices[prices.index <= date]
                        # Optimize weights
                        new_weights = strategy.optimize(historical_returns, prices=historical_prices, **kwargs)
                        
                        # Rebalance portfolio
                        portfolio.rebalance(new_weights, current_prices)
                
                rebalance_index += 1
        
        # Calculate performance metrics
        portfolio_series = portfolio.get_portfolio_series()
        metrics = PortfolioMetrics(self.risk_free_rate)
        
        # Calculate all metrics
        performance_metrics = metrics.calculate_all_metrics(portfolio_series)
        
        # Calculate rolling metrics only at rebalance dates
        # Use smaller window if we have limited data points
        data_length = len(portfolio_series)
        if data_length < 252:
            window_size = max(10, data_length // 4)  # Use 1/4 of data length, minimum 10
        else:
            window_size = 252
        
        # Calculate rolling metrics for the full series first
        full_rolling_metrics = metrics.calculate_rolling_metrics(portfolio_series, window=window_size)
        
        # Filter to only include rebalance dates
        if not full_rolling_metrics.empty and rebalance_dates:
            # Convert rebalance_dates to datetime if they're strings
            rebalance_dates_dt = [pd.to_datetime(date) if isinstance(date, str) else date for date in rebalance_dates]
            
            # Filter rolling metrics to only include rebalance dates
            available_dates = [date for date in rebalance_dates_dt if date in full_rolling_metrics.index]
            rolling_metrics = full_rolling_metrics.loc[available_dates]
        else:
            rolling_metrics = full_rolling_metrics
        
        # Compile results
        results = {
            'strategy_name': strategy.name,
            'strategy_params': strategy.get_parameters(),
            'portfolio_values': portfolio_series,
            'weight_history': portfolio.get_weight_history(),
            'performance_metrics': performance_metrics,
            'rolling_metrics': rolling_metrics,
            'transaction_history': portfolio.transaction_history,
            'summary': portfolio.get_summary(),
            'rebalance_dates': rebalance_dates,
            'last_weights': self.renormalize_last_weights(portfolio),
            'backtest_params': {
                'initial_capital': self.initial_capital,
                'risk_free_rate': self.risk_free_rate,
                'transaction_costs': self.transaction_costs,
                'rebalance_freq': kwargs.get('rebalance_freq', {}),
                'start_date': prices.index.min(),
                'end_date': prices.index.max()
            }
        }
        
        # Calculate significance test after results are compiled
        significance_results = self._calculate_significance(portfolio_series)
        results['significance'] = significance_results
        
        return results
    
    def renormalize_last_weights(self, portfolio: Portfolio, min_weight: float = 0.01) -> Dict[str, float]:
        """Get the most recent set of weights with minimum weight of 0.01 and renormalized.
        
        Args:
            portfolio: Portfolio instance containing weight history
            
        Returns:
            Dictionary of renormalized weights with minimum weight of 0.01, 
            rounded to nearest 0.01 and sorted by weight in descending order
        """
        # Get the most recent weights from weight history
        if not portfolio.weight_history:
            return {}
        
        # Get the last weight allocation
        last_weights = portfolio.weight_history[-1].copy()
        
        if not last_weights:
            return {}
        
        # Apply minimum weight constraint of 0.01
        normalized_weights = {}
        for ticker, weight in last_weights.items():
            normalized_weights[ticker] = max(weight, min_weight)
        
        # Renormalize to sum to 1.0
        total_weight = sum(normalized_weights.values())
        if total_weight > 0:
            for ticker in normalized_weights:
                normalized_weights[ticker] /= total_weight
        
        # Verify minimum weight constraint is satisfied after renormalization
        # If not, we need to iteratively adjust
        min_weight = min(normalized_weights.values())
        if min_weight < min_weight:
            # Iteratively adjust weights to ensure minimum constraint
            while min_weight < min_weight:
                # Find weights below minimum and set them to minimum
                for ticker in normalized_weights:
                    if normalized_weights[ticker] < min_weight:
                        normalized_weights[ticker] = min_weight
                
                # Renormalize again
                total_weight = sum(normalized_weights.values())
                if total_weight > 0:
                    for ticker in normalized_weights:
                        normalized_weights[ticker] /= total_weight
                
                min_weight = min(normalized_weights.values())
        
        # Round all weights to the nearest 0.01
        rounded_weights = {}
        for ticker, weight in normalized_weights.items():
            rounded_weights[ticker] = round(weight, 2)
        
        # Adjust rounding to ensure weights sum to 1.0 while maintaining minimum constraint
        total_rounded = sum(rounded_weights.values())
        if abs(total_rounded - 1.0) > 1e-6:
            # Sort weights by value to find the best candidates for adjustment
            sorted_items = sorted(rounded_weights.items(), key=lambda x: x[1], reverse=True)
            
            if total_rounded > 1.0:
                # Need to reduce weights - start from the largest
                excess = total_rounded - 1.0
                for ticker, weight in sorted_items:
                    if excess <= 0:
                        break
                    reduction = min(excess, weight - min_weight)  # Don't go below minimum
                    if reduction > 0:
                        rounded_weights[ticker] = round(weight - reduction, 2)
                        excess -= reduction
            else:
                # Need to increase weights - start from the largest
                deficit = 1.0 - total_rounded
                for ticker, weight in sorted_items:
                    if deficit <= 0:
                        break
                    increase = min(deficit, 1.0 - weight)  # Don't go above 0.99
                    if increase > 0:
                        rounded_weights[ticker] = round(weight + increase, 2)
                        deficit -= increase
        
        # Sort by weight in descending order
        sorted_weights = dict(sorted(rounded_weights.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True))
        
        return sorted_weights
    
    def _calculate_significance(self, portfolio_series: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate significance test for portfolio returns.
        
        Args:
            portfolio_series: Portfolio value series
            benchmark_returns: Optional benchmark returns for relative performance test
            
        Returns:
            Dictionary containing t-statistic and p-value
        """
        # Calculate portfolio returns
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        if len(portfolio_returns) < 2:
            return {'t_statistic': np.nan, 'p_value': np.nan}
        
        if benchmark_returns is not None:
            # Test if strategy outperforms benchmark
            # Align dates
            aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
            if len(aligned_returns) < 2:
                return {'t_statistic': np.nan, 'p_value': np.nan}
            
            strategy_returns = aligned_returns.iloc[:, 0]
            benchmark_returns_aligned = aligned_returns.iloc[:, 1]
            
            # Calculate excess returns
            excess_returns = strategy_returns - benchmark_returns_aligned
            
            # One-sided t-test: H0: mean <= 0, H1: mean > 0
            t_statistic, p_value = stats.ttest_1samp(excess_returns, 0, alternative='greater')
        else:
            # Test if strategy returns are significantly greater than zero
            # One-sided t-test: H0: mean <= 0, H1: mean > 0
            t_statistic, p_value = stats.ttest_1samp(portfolio_returns, 0, alternative='greater')
        
        return {
            't_statistic': t_statistic,
            'p_value': p_value
        }
    
    def significance(self, strategy_name: str, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Perform a one-sided t-test to test if strategy returns are significantly greater than zero.
        
        Args:
            strategy_name: Name of the strategy to test
            benchmark_returns: Optional benchmark returns for relative performance test
            
        Returns:
            Dictionary containing t-statistic and p-value
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy '{strategy_name}' not found in results")
        
        # Get portfolio values and calculate significance
        portfolio_series = self.results[strategy_name]['portfolio_values']
        return self._calculate_significance(portfolio_series, benchmark_returns)
    
    def get_portfolio_returns(self, strategy_name: str) -> pd.Series:
        """Get portfolio returns for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Series containing portfolio returns (percentage change)
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy '{strategy_name}' not found in results")
        
        # Get portfolio values and calculate returns
        portfolio_series = self.results[strategy_name]['portfolio_values']
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        return portfolio_returns
    
    def compare_strategies(self, strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare performance of different strategies.
        
        Args:
            strategy_names: List of strategy names to compare (if None, compares all)
            
        Returns:
            DataFrame with comparison metrics
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        comparison_data = []
        
        for strategy_name in strategy_names:
            if strategy_name not in self.results:
                continue
            
            results = self.results[strategy_name]
            metrics = results['performance_metrics']
            
            # Get significance test results
            significance = results.get('significance', {})
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': metrics.get('total_return', 0) * 100,
                'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                'Volatility (%)': metrics.get('volatility', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                'Calmar Ratio': metrics.get('calmar_ratio', 0),
                'VaR (5%)': metrics.get('var_5pct', 0),
                'CVaR (5%)': metrics.get('cvar_5pct', 0),
                'T-Statistic': significance.get('t_statistic', np.nan),
                'P-Value': significance.get('p_value', np.nan)
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_strategy_results(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy results or None if not found
        """
        return self.results.get(strategy_name)
    
    def plot_portfolio_values(self, strategy_names: Optional[List[str]] = None) -> None:
        """Plot portfolio values for different strategies."""
        import matplotlib.pyplot as plt
        
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        plt.figure(figsize=(12, 8))
        
        for strategy_name in strategy_names:
            if strategy_name in self.results:
                portfolio_series = self.results[strategy_name]['portfolio_values']
                plt.plot(portfolio_series.index, portfolio_series.values, 
                        label=strategy_name, linewidth=2)
        
        plt.title('Portfolio Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename: str, format: str = 'csv') -> None:
        """Export backtest results to file.
        
        Args:
            filename: Output filename
            format: Export format ('csv', 'excel', 'json')
        """
        if format == 'csv':
            comparison_df = self.compare_strategies()
            comparison_df.to_csv(filename, index=False)
        elif format == 'excel':
            comparison_df = self.compare_strategies()
            comparison_df.to_excel(filename, index=False)
        elif format == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(self.results, f, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to {filename}")
