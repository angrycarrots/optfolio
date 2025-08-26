"""Portfolio performance metrics calculation."""

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats


class PortfolioMetrics:
    """Calculate portfolio performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize portfolio metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate portfolio returns.
        
        Args:
            portfolio_values: Series of portfolio values over time
            
        Returns:
            Series of returns
        """
        return portfolio_values.pct_change(fill_method=None).dropna()
    
    def calculate_cumulative_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate cumulative returns.
        
        Args:
            portfolio_values: Series of portfolio values over time
            
        Returns:
            Series of cumulative returns
        """
        return (portfolio_values / portfolio_values.iloc[0]) - 1
    
    def calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series.
        
        Args:
            portfolio_values: Series of portfolio values over time
            
        Returns:
            Series of drawdown percentages
        """
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            portfolio_values: Series of portfolio values over time
            
        Returns:
            Maximum drawdown as a percentage
        """
        drawdown = self.calculate_drawdown(portfolio_values)
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return np.nan
            
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return np.nan
            
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if excess_returns.mean() > 0 else -np.inf
            
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
    
    def calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate portfolio beta relative to market.
        
        Args:
            portfolio_returns: Series of portfolio returns
            market_returns: Series of market returns
            
        Returns:
            Beta coefficient
        """
        if len(portfolio_returns) != len(market_returns):
            # Align the series
            common_index = portfolio_returns.index.intersection(market_returns.index)
            portfolio_returns = portfolio_returns.loc[common_index]
            market_returns = market_returns.loc[common_index]
            
        if len(portfolio_returns) < 2:
            return np.nan
            
        # Remove any NaN values
        valid_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'market': market_returns
        }).dropna()
        
        if len(valid_data) < 2:
            return np.nan
            
        covariance = np.cov(valid_data['portfolio'], valid_data['market'])[0, 1]
        market_variance = np.var(valid_data['market'])
        
        return covariance / market_variance if market_variance != 0 else np.nan
    
    def calculate_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Annualized volatility
        """
        if len(returns) < 2:
            return np.nan
            
        return returns.std() * np.sqrt(periods_per_year)
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default: 5%)
            
        Returns:
            Value at Risk
        """
        if len(returns) == 0:
            return np.nan
            
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default: 5%)
            
        Returns:
            Conditional Value at Risk
        """
        if len(returns) == 0:
            return np.nan
            
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_calmar_ratio(self, portfolio_values: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            portfolio_values: Series of portfolio values over time
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Calmar ratio
        """
        if len(portfolio_values) < 2:
            return np.nan
            
        returns = self.calculate_returns(portfolio_values)
        annual_return = returns.mean() * periods_per_year
        max_dd = abs(self.calculate_max_drawdown(portfolio_values))
        
        return annual_return / max_dd if max_dd != 0 else np.nan
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio.
        
        Args:
            portfolio_returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Information ratio
        """
        if len(portfolio_returns) != len(benchmark_returns):
            # Align the series
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]
            
        if len(portfolio_returns) < 2:
            return np.nan
            
        # Remove any NaN values
        valid_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(valid_data) < 2:
            return np.nan
            
        active_returns = valid_data['portfolio'] - valid_data['benchmark']
        return active_returns.mean() / active_returns.std()
    
    def calculate_treynor_ratio(self, portfolio_returns: pd.Series, market_returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Treynor ratio.
        
        Args:
            portfolio_returns: Series of portfolio returns
            market_returns: Series of market returns
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Treynor ratio
        """
        if len(portfolio_returns) < 2:
            return np.nan
            
        beta = self.calculate_beta(portfolio_returns, market_returns)
        if np.isnan(beta) or beta == 0:
            return np.nan
            
        excess_return = portfolio_returns.mean() - (self.risk_free_rate / periods_per_year)
        return (excess_return * periods_per_year) / beta
    
    def calculate_all_metrics(self, portfolio_values: pd.Series, 
                            market_returns: Optional[pd.Series] = None,
                            benchmark_returns: Optional[pd.Series] = None,
                            periods_per_year: int = 252) -> Dict[str, float]:
        """Calculate all portfolio metrics.
        
        Args:
            portfolio_values: Series of portfolio values over time
            market_returns: Series of market returns (for beta, Treynor ratio)
            benchmark_returns: Series of benchmark returns (for information ratio)
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        returns = self.calculate_returns(portfolio_values)
        
        # Calculate time period in years
        total_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        years = total_days / 365.25
        
        total_return = self.calculate_cumulative_returns(portfolio_values).iloc[-1]
        
        # Calculate annualized return using compound annual growth rate (CAGR)
        if years > 0 and total_return > -1:  # Avoid division by zero and negative total return
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': self.calculate_volatility(returns, periods_per_year),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, periods_per_year),
            'sortino_ratio': self.calculate_sortino_ratio(returns, periods_per_year),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values),
            'calmar_ratio': self.calculate_calmar_ratio(portfolio_values, periods_per_year),
            'var_5pct': self.calculate_var(returns, 0.05),
            'cvar_5pct': self.calculate_cvar(returns, 0.05),
        }
        
        if market_returns is not None:
            metrics['beta'] = self.calculate_beta(returns, market_returns)
            metrics['treynor_ratio'] = self.calculate_treynor_ratio(returns, market_returns, periods_per_year)
            
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
            
        return metrics
    
    def calculate_rolling_metrics(self, portfolio_values: pd.Series, 
                                window: int = 252,
                                market_returns: Optional[pd.Series] = None,
                                periods_per_year: int = 252) -> pd.DataFrame:
        """Calculate rolling portfolio metrics.
        
        Args:
            portfolio_values: Series of portfolio values over time
            window: Rolling window size in periods
            market_returns: Series of market returns (for beta)
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            DataFrame with rolling metrics
        """
        # Validate inputs
        if len(portfolio_values) < 2:
            return pd.DataFrame()
        
        # Adjust window size if it's too large for the data
        if window > len(portfolio_values):
            window = max(2, len(portfolio_values) // 2)
        
        returns = self.calculate_returns(portfolio_values)
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window, min_periods=1).mean() * periods_per_year
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window, min_periods=2).std() * np.sqrt(periods_per_year)
        
        # Rolling Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        rolling_vol = returns.rolling(window, min_periods=2).std()
        rolling_mean = excess_returns.rolling(window, min_periods=1).mean()
        
        # Calculate Sharpe ratio only where volatility is not zero
        rolling_metrics['rolling_sharpe'] = np.where(
            rolling_vol > 0,
            (rolling_mean / rolling_vol) * np.sqrt(periods_per_year),
            np.nan
        )
        
        # Rolling drawdown
        drawdown = self.calculate_drawdown(portfolio_values)
        rolling_metrics['rolling_drawdown'] = drawdown.rolling(window, min_periods=1).min()
        
        # Rolling beta (if market returns provided)
        if market_returns is not None and len(market_returns) >= window:
            rolling_beta = []
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                window_market = market_returns.iloc[i-window:i]
                beta = self.calculate_beta(window_returns, window_market)
                rolling_beta.append(beta)
            
            # Pad with NaN values
            rolling_beta = [np.nan] * (window - 1) + rolling_beta
            rolling_metrics['rolling_beta'] = rolling_beta
            
        return rolling_metrics
