"""Tests for portfolio metrics calculation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..portfolio.metrics import PortfolioMetrics


class TestPortfolioMetrics:
    """Test cases for PortfolioMetrics class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample portfolio values
        dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create a portfolio that grows from 100 to 200 over 3 years (100% total return)
        self.portfolio_values = pd.Series(100, index=dates)
        self.portfolio_values.iloc[-1] = 200
        
        # Create a more realistic portfolio with some volatility
        returns = np.random.normal(0.001, 0.02, len(dates))
        self.volatile_portfolio = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        
        # Create a declining portfolio
        self.declining_portfolio = pd.Series(100 * np.exp(-0.0005 * np.arange(len(dates))), index=dates)
        
        self.metrics = PortfolioMetrics(risk_free_rate=0.02)
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        returns = self.metrics.calculate_returns(self.portfolio_values)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(self.portfolio_values) - 1  # One less due to pct_change
        assert returns.index.equals(self.portfolio_values.index[1:])
        
        # Test that the first return is approximately 0 (no change from initial value)
        assert abs(returns.iloc[0]) < 1e-10
    
    def test_calculate_cumulative_returns(self):
        """Test cumulative returns calculation."""
        cumulative_returns = self.metrics.calculate_cumulative_returns(self.portfolio_values)
        
        assert isinstance(cumulative_returns, pd.Series)
        assert len(cumulative_returns) == len(self.portfolio_values)
        assert cumulative_returns.iloc[0] == 0.0  # Start at 0
        assert abs(cumulative_returns.iloc[-1] - 1.0) < 1e-10  # End at 100% (200/100 - 1)
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        drawdown = self.metrics.calculate_drawdown(self.volatile_portfolio)
        
        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(self.volatile_portfolio)
        assert drawdown.iloc[0] == 0.0  # First value has no drawdown
        assert all(drawdown <= 0)  # Drawdown should be non-positive
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        max_drawdown = self.metrics.calculate_max_drawdown(self.volatile_portfolio)
        
        assert isinstance(max_drawdown, float)
        assert max_drawdown <= 0  # Should be non-positive
        assert max_drawdown >= -1  # Should not be less than -100%
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        sharpe = self.metrics.calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # Test with insufficient data
        short_returns = returns.iloc[:1]
        sharpe_short = self.metrics.calculate_sharpe_ratio(short_returns)
        assert np.isnan(sharpe_short)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        sortino = self.metrics.calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        
        # Test with all positive returns
        positive_returns = pd.Series([0.01, 0.02, 0.01], index=pd.date_range('2021-01-01', periods=3))
        sortino_positive = self.metrics.calculate_sortino_ratio(positive_returns)
        assert sortino_positive == np.inf  # Should be infinity for all positive returns
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        volatility = self.metrics.calculate_volatility(returns)
        
        assert isinstance(volatility, float)
        assert volatility > 0
        assert not np.isnan(volatility)
    
    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        var_5pct = self.metrics.calculate_var(returns, 0.05)
        
        assert isinstance(var_5pct, float)
        assert var_5pct < 0  # VaR should be negative (loss)
        assert not np.isnan(var_5pct)
    
    def test_calculate_cvar(self):
        """Test Conditional Value at Risk calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        cvar_5pct = self.metrics.calculate_cvar(returns, 0.05)
        
        assert isinstance(cvar_5pct, float)
        assert cvar_5pct < 0  # CVaR should be negative (loss)
        assert not np.isnan(cvar_5pct)
        assert cvar_5pct <= self.metrics.calculate_var(returns, 0.05)  # CVaR <= VaR
    
    def test_calculate_beta(self):
        """Test beta calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        market_returns = pd.Series(np.random.normal(0.001, 0.015, len(returns)), index=returns.index)
        
        beta = self.metrics.calculate_beta(returns, market_returns)
        
        assert isinstance(beta, float)
        assert not np.isnan(beta)
    
    def test_calculate_information_ratio(self):
        """Test information ratio calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        benchmark_returns = pd.Series(np.random.normal(0.001, 0.015, len(returns)), index=returns.index)
        
        info_ratio = self.metrics.calculate_information_ratio(returns, benchmark_returns)
        
        assert isinstance(info_ratio, float)
        assert not np.isnan(info_ratio)
    
    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calmar = self.metrics.calculate_calmar_ratio(self.volatile_portfolio)
        
        assert isinstance(calmar, float)
        assert not np.isnan(calmar)
    
    def test_calculate_treynor_ratio(self):
        """Test Treynor ratio calculation."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        market_returns = pd.Series(np.random.normal(0.001, 0.015, len(returns)), index=returns.index)
        
        treynor = self.metrics.calculate_treynor_ratio(returns, market_returns)
        
        assert isinstance(treynor, float)
        assert not np.isnan(treynor)
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        metrics_dict = self.metrics.calculate_all_metrics(self.volatile_portfolio)
        
        expected_keys = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'var_5pct', 'cvar_5pct'
        ]
        
        for key in expected_keys:
            assert key in metrics_dict
            assert isinstance(metrics_dict[key], (int, float))
            assert not np.isnan(metrics_dict[key])
    
    def test_annualized_return_calculation(self):
        """Test that annualized return calculation is correct."""
        # Test with simple case: portfolio doubles over 3 years
        metrics_dict = self.metrics.calculate_all_metrics(self.portfolio_values)
        
        total_return = metrics_dict['total_return']
        annualized_return = metrics_dict['annualized_return']
        
        # Total return should be 100% (doubles from 100 to 200)
        assert abs(total_return - 1.0) < 1e-10
        
        # Annualized return should be approximately 26% (cube root of 2 - 1)
        expected_annualized = (1 + total_return) ** (1/3) - 1
        assert abs(annualized_return - expected_annualized) < 0.001
        
        # Annualized return should be less than total return for periods > 1 year
        assert annualized_return < total_return
    
    def test_annualized_return_edge_cases(self):
        """Test annualized return calculation with edge cases."""
        # Test with very short period
        short_dates = pd.date_range('2021-01-01', '2021-01-02', freq='D')
        short_portfolio = pd.Series([100, 101], index=short_dates)
        
        metrics_short = self.metrics.calculate_all_metrics(short_portfolio)
        # For a 1-day period with 1% return, annualized return should be very high
        assert metrics_short['annualized_return'] > 0  # Should be positive for positive return
        
        # Test with declining portfolio
        metrics_declining = self.metrics.calculate_all_metrics(self.declining_portfolio)
        assert metrics_declining['annualized_return'] < 0  # Should be negative
        
        # Test with zero return
        zero_return_dates = pd.date_range('2021-01-01', '2021-01-02', freq='D')
        zero_return_portfolio = pd.Series([100, 100], index=zero_return_dates)
        metrics_zero = self.metrics.calculate_all_metrics(zero_return_portfolio)
        assert metrics_zero['annualized_return'] == 0.0  # Should be zero for no return
    
    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation."""
        rolling_metrics = self.metrics.calculate_rolling_metrics(self.volatile_portfolio, window=60)
        
        assert isinstance(rolling_metrics, pd.DataFrame)
        assert not rolling_metrics.empty
        
        expected_columns = ['rolling_return', 'rolling_volatility', 'rolling_sharpe', 'rolling_drawdown']
        for col in expected_columns:
            assert col in rolling_metrics.columns
        
        # Test with window larger than data
        large_window_metrics = self.metrics.calculate_rolling_metrics(self.volatile_portfolio, window=10000)
        assert not large_window_metrics.empty
    
    def test_calculate_rolling_metrics_edge_cases(self):
        """Test rolling metrics with edge cases."""
        # Test with very short series
        short_series = pd.Series([100, 101, 102], index=pd.date_range('2021-01-01', periods=3))
        rolling_short = self.metrics.calculate_rolling_metrics(short_series, window=2)
        assert not rolling_short.empty
        
        # Test with constant values
        constant_series = pd.Series(100, index=pd.date_range('2021-01-01', periods=100))
        rolling_constant = self.metrics.calculate_rolling_metrics(constant_series, window=20)
        assert rolling_constant['rolling_volatility'].mean() == 0.0
    
    def test_risk_free_rate_impact(self):
        """Test that risk-free rate affects Sharpe and Sortino ratios."""
        returns = self.metrics.calculate_returns(self.volatile_portfolio)
        
        # Test with different risk-free rates
        metrics_low_rf = PortfolioMetrics(risk_free_rate=0.01)
        metrics_high_rf = PortfolioMetrics(risk_free_rate=0.05)
        
        sharpe_low = metrics_low_rf.calculate_sharpe_ratio(returns)
        sharpe_high = metrics_high_rf.calculate_sharpe_ratio(returns)
        
        # Higher risk-free rate should result in lower Sharpe ratio
        assert sharpe_low > sharpe_high


class TestPortfolioMetricsIntegration:
    """Integration tests for portfolio metrics."""
    
    def test_metrics_consistency(self):
        """Test that different metrics are internally consistent."""
        dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create a portfolio with known characteristics
        returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_values = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        
        metrics = PortfolioMetrics()
        metrics_dict = metrics.calculate_all_metrics(portfolio_values)
        
        # Test internal consistency
        assert metrics_dict['total_return'] >= metrics_dict['max_drawdown']  # Total return should exceed max drawdown
        assert metrics_dict['volatility'] > 0  # Volatility should be positive
        assert metrics_dict['annualized_return'] < metrics_dict['total_return']  # For periods > 1 year
        
        # Test that Sharpe and Sortino ratios are reasonable
        assert -10 < metrics_dict['sharpe_ratio'] < 10
        assert -10 < metrics_dict['sortino_ratio'] < 10
