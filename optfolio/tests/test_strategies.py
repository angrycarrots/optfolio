"""Tests for portfolio optimization strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..strategies.base import OptimizationStrategy, StrategyFactory
from ..strategies.equal_weight import EqualWeightStrategy
from ..strategies.mean_variance import MeanVarianceStrategy
from ..strategies.random_weight import RandomWeightStrategy
from ..strategies.black_litterman import BlackLittermanStrategy


class TestStrategyBase:
    """Test cases for base strategy class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample returns data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.001, 0.025, len(dates)),
            'MSFT': np.random.normal(0.0008, 0.018, len(dates)),
            'TSLA': np.random.normal(0.002, 0.04, len(dates))
        }, index=dates)
    
    def test_strategy_validation(self):
        """Test strategy input validation."""
        # Create a mock strategy
        class MockStrategy(OptimizationStrategy):
            def optimize(self, returns, **kwargs):
                return {'AAPL': 0.5, 'GOOGL': 0.5}
            
            def get_parameters(self):
                return {'test': True}
        
        strategy = MockStrategy("Test Strategy")
        
        # Test valid inputs
        strategy.validate_inputs(self.returns)
        
        # Test empty DataFrame
        with pytest.raises(ValueError, match="Returns DataFrame is empty"):
            strategy.validate_inputs(pd.DataFrame())
        
        # Test wrong index type
        wrong_index_returns = self.returns.copy()
        wrong_index_returns.index = range(len(wrong_index_returns))
        
        with pytest.raises(ValueError, match="Returns DataFrame must have DatetimeIndex"):
            strategy.validate_inputs(wrong_index_returns)
    
    def test_preprocess_returns(self):
        """Test returns preprocessing."""
        # Create a mock strategy
        class MockStrategy(OptimizationStrategy):
            def optimize(self, returns, **kwargs):
                return {'AAPL': 0.5, 'GOOGL': 0.5}
            
            def get_parameters(self):
                return {'test': True}
        
        strategy = MockStrategy("Test Strategy")
        
        # Add some missing values
        returns_with_nan = self.returns.copy()
        returns_with_nan.loc['2021-01-01', 'AAPL'] = np.nan
        
        # Test forward fill
        processed = strategy.preprocess_returns(returns_with_nan, method="forward_fill")
        assert not processed.isnull().any().any()
        
        # Test drop
        processed = strategy.preprocess_returns(returns_with_nan, method="drop")
        assert len(processed) < len(returns_with_nan)
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown preprocessing method"):
            strategy.preprocess_returns(returns_with_nan, method="invalid")
    
    def test_calculate_expected_returns(self):
        """Test expected returns calculation."""
        class MockStrategy(OptimizationStrategy):
            def optimize(self, returns, **kwargs):
                return {'AAPL': 0.5, 'GOOGL': 0.5}
            
            def get_parameters(self):
                return {'test': True}
        
        strategy = MockStrategy("Test Strategy")
        
        # Test mean method
        expected_returns = strategy.calculate_expected_returns(self.returns, method="mean")
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == len(self.returns.columns)
        
        # Test geometric mean
        expected_returns = strategy.calculate_expected_returns(self.returns, method="geometric_mean")
        assert isinstance(expected_returns, pd.Series)
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown expected returns method"):
            strategy.calculate_expected_returns(self.returns, method="invalid")
    
    def test_calculate_covariance_matrix(self):
        """Test covariance matrix calculation."""
        class MockStrategy(OptimizationStrategy):
            def optimize(self, returns, **kwargs):
                return {'AAPL': 0.5, 'GOOGL': 0.5}
            
            def get_parameters(self):
                return {'test': True}
        
        strategy = MockStrategy("Test Strategy")
        
        # Test sample covariance
        cov = strategy.calculate_covariance_matrix(self.returns, method="sample")
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (len(self.returns.columns), len(self.returns.columns))
        
        # Test exponential covariance
        cov = strategy.calculate_covariance_matrix(self.returns, method="exponential")
        assert isinstance(cov, pd.DataFrame)
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown covariance method"):
            strategy.calculate_covariance_matrix(self.returns, method="invalid")


class TestEqualWeightStrategy:
    """Test cases for Equal Weight strategy."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.001, 0.025, len(dates)),
            'MSFT': np.random.normal(0.0008, 0.018, len(dates)),
            'TSLA': np.random.normal(0.002, 0.04, len(dates))
        }, index=dates)
    
    def test_equal_weight_optimization(self):
        """Test equal weight optimization."""
        strategy = EqualWeightStrategy()
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        
        # Check that weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # Check that all weights are equal
        expected_weight = 1.0 / len(self.returns.columns)
        for weight in weights.values():
            assert abs(weight - expected_weight) < 1e-6
    
    def test_equal_weight_with_constraints(self):
        """Test equal weight with constraints."""
        strategy = EqualWeightStrategy()
        constraints = {'min_weight': 0.1, 'max_weight': 0.4}
        
        weights = strategy.optimize(self.returns, constraints=constraints)
        
        assert isinstance(weights, dict)
        
        # Check constraints
        for weight in weights.values():
            assert weight >= 0.1
            assert weight <= 0.4
    
    def test_equal_weight_parameters(self):
        """Test equal weight strategy parameters."""
        strategy = EqualWeightStrategy()
        params = strategy.get_parameters()
        
        assert isinstance(params, dict)
        assert params['strategy_type'] == 'equal_weight'
        assert 'description' in params


class TestMeanVarianceStrategy:
    """Test cases for Mean-Variance strategy."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.001, 0.025, len(dates)),
            'MSFT': np.random.normal(0.0008, 0.018, len(dates)),
            'TSLA': np.random.normal(0.002, 0.04, len(dates))
        }, index=dates)
    
    def test_mean_variance_sortino_optimization(self):
        """Test mean-variance optimization with Sortino ratio objective."""
        strategy = MeanVarianceStrategy(objective="sortino_ratio")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_mean_variance_sharpe_optimization(self):
        """Test mean-variance optimization with Sharpe ratio objective."""
        strategy = MeanVarianceStrategy(objective="sharpe_ratio")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_mean_variance_min_variance_optimization(self):
        """Test mean-variance optimization with minimum variance objective."""
        strategy = MeanVarianceStrategy(objective="min_variance")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_mean_variance_max_return_optimization(self):
        """Test mean-variance optimization with maximum return objective."""
        strategy = MeanVarianceStrategy(objective="max_return")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_mean_variance_invalid_objective(self):
        """Test mean-variance optimization with invalid objective."""
        strategy = MeanVarianceStrategy(objective="invalid")
        
        with pytest.raises(ValueError, match="Unknown objective"):
            strategy.optimize(self.returns)
    
    def test_mean_variance_parameters(self):
        """Test mean-variance strategy parameters."""
        strategy = MeanVarianceStrategy(objective="sortino_ratio")
        params = strategy.get_parameters()
        
        assert isinstance(params, dict)
        assert params['strategy_type'] == 'mean_variance'
        assert params['objective'] == 'sortino_ratio'
        assert 'risk_free_rate' in params


class TestRandomWeightStrategy:
    """Test cases for Random Weight strategy."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.001, 0.025, len(dates)),
            'MSFT': np.random.normal(0.0008, 0.018, len(dates)),
            'TSLA': np.random.normal(0.002, 0.04, len(dates))
        }, index=dates)
    
    def test_random_weight_dirichlet_optimization(self):
        """Test random weight optimization with Dirichlet distribution."""
        strategy = RandomWeightStrategy(distribution="dirichlet", seed=42)
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_random_weight_uniform_optimization(self):
        """Test random weight optimization with uniform distribution."""
        strategy = RandomWeightStrategy(distribution="uniform", seed=42)
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_random_weight_normal_optimization(self):
        """Test random weight optimization with normal distribution."""
        strategy = RandomWeightStrategy(distribution="normal", seed=42)
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_random_weight_invalid_distribution(self):
        """Test random weight optimization with invalid distribution."""
        strategy = RandomWeightStrategy(distribution="invalid")
        
        with pytest.raises(ValueError, match="Unknown distribution"):
            strategy.optimize(self.returns)
    
    def test_random_weight_reproducibility(self):
        """Test random weight reproducibility with seed."""
        strategy1 = RandomWeightStrategy(seed=42)
        strategy2 = RandomWeightStrategy(seed=42)
        
        weights1 = strategy1.optimize(self.returns)
        weights2 = strategy2.optimize(self.returns)
        
        # Weights should be identical with same seed
        for ticker in weights1:
            assert abs(weights1[ticker] - weights2[ticker]) < 1e-6
    
    def test_generate_multiple_weights(self):
        """Test generating multiple random portfolios."""
        strategy = RandomWeightStrategy(seed=42)
        portfolios = strategy.generate_multiple_weights(self.returns, n_portfolios=5)
        
        assert isinstance(portfolios, list)
        assert len(portfolios) == 5
        
        for portfolio in portfolios:
            assert isinstance(portfolio, dict)
            assert len(portfolio) == len(self.returns.columns)
            assert abs(sum(portfolio.values()) - 1.0) < 1e-6
    
    def test_random_weight_parameters(self):
        """Test random weight strategy parameters."""
        strategy = RandomWeightStrategy(distribution="dirichlet", seed=42)
        params = strategy.get_parameters()
        
        assert isinstance(params, dict)
        assert params['strategy_type'] == 'random_weight'
        assert params['distribution'] == 'dirichlet'
        assert params['seed'] == 42


class TestBlackLittermanStrategy:
    """Test cases for Black-Litterman strategy."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(0.001, 0.025, len(dates)),
            'MSFT': np.random.normal(0.0008, 0.018, len(dates)),
            'TSLA': np.random.normal(0.002, 0.04, len(dates))
        }, index=dates)
    
    def test_black_litterman_market_cap_priors(self):
        """Test Black-Litterman with market cap priors."""
        strategy = BlackLittermanStrategy(prior_method="market_cap", view_method="random")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_black_litterman_equal_priors(self):
        """Test Black-Litterman with equal priors."""
        strategy = BlackLittermanStrategy(prior_method="equal", view_method="random")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_black_litterman_momentum_views(self):
        """Test Black-Litterman with momentum views."""
        strategy = BlackLittermanStrategy(prior_method="market_cap", view_method="momentum")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_black_litterman_mean_reversion_views(self):
        """Test Black-Litterman with mean reversion views."""
        strategy = BlackLittermanStrategy(prior_method="market_cap", view_method="mean_reversion")
        weights = strategy.optimize(self.returns)
        
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_black_litterman_invalid_prior_method(self):
        """Test Black-Litterman with invalid prior method."""
        strategy = BlackLittermanStrategy(prior_method="invalid", view_method="random")
        
        with pytest.raises(ValueError, match="Unknown prior method"):
            strategy.optimize(self.returns)
    
    def test_black_litterman_invalid_view_method(self):
        """Test Black-Litterman with invalid view method."""
        strategy = BlackLittermanStrategy(prior_method="market_cap", view_method="invalid")
        
        with pytest.raises(ValueError, match="Unknown view method"):
            strategy.optimize(self.returns)
    
    def test_black_litterman_parameters(self):
        """Test Black-Litterman strategy parameters."""
        strategy = BlackLittermanStrategy(
            tau=0.05, 
            risk_aversion=2.5,
            prior_method="market_cap",
            view_method="random"
        )
        params = strategy.get_parameters()
        
        assert isinstance(params, dict)
        assert params['strategy_type'] == 'black_litterman'
        assert params['tau'] == 0.05
        assert params['risk_aversion'] == 2.5
        assert params['prior_method'] == 'market_cap'
        assert params['view_method'] == 'random'


class TestStrategyFactory:
    """Test cases for StrategyFactory."""
    
    def test_strategy_registration(self):
        """Test strategy registration."""
        # Test that strategies are registered
        available_strategies = StrategyFactory.list_strategies()
        
        assert 'equal_weight' in available_strategies
        assert 'mean_variance' in available_strategies
        assert 'random_weight' in available_strategies
        assert 'black_litterman' in available_strategies
    
    def test_create_equal_weight_strategy(self):
        """Test creating equal weight strategy."""
        strategy = StrategyFactory.create('equal_weight')
        strategy.name = "Test Equal Weight"
        
        assert isinstance(strategy, EqualWeightStrategy)
        assert strategy.name == "Test Equal Weight"
    
    def test_create_mean_variance_strategy(self):
        """Test creating mean-variance strategy."""
        strategy = StrategyFactory.create('mean_variance',
                                        objective="sortino_ratio")
        strategy.name = "Test MV"
        
        assert isinstance(strategy, MeanVarianceStrategy)
        assert strategy.name == "Test MV"
        assert strategy.objective == "sortino_ratio"
    
    def test_create_random_weight_strategy(self):
        """Test creating random weight strategy."""
        strategy = StrategyFactory.create('random_weight',
                                        distribution="dirichlet",
                                        seed=42)
        strategy.name = "Test Random"
        
        assert isinstance(strategy, RandomWeightStrategy)
        assert strategy.name == "Test Random"
        assert strategy.distribution == "dirichlet"
        assert strategy.seed == 42
    
    def test_create_black_litterman_strategy(self):
        """Test creating Black-Litterman strategy."""
        strategy = StrategyFactory.create('black_litterman',
                                        prior_method="market_cap",
                                        view_method="random")
        strategy.name = "Test BL"
        
        assert isinstance(strategy, BlackLittermanStrategy)
        assert strategy.name == "Test BL"
        assert strategy.prior_method == "market_cap"
        assert strategy.view_method == "random"
    
    def test_create_invalid_strategy(self):
        """Test creating invalid strategy."""
        with pytest.raises(ValueError, match="Strategy 'invalid' not found"):
            StrategyFactory.create('invalid')
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = StrategyFactory.list_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) >= 4  # Should have at least our 4 strategies
        assert all(isinstance(s, str) for s in strategies)
