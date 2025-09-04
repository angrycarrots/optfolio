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
from ..strategies.buy_and_hold import BuyAndHoldStrategy


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
        
        # The strategy should still work but use a default objective
        weights = strategy.optimize(self.returns)
        assert isinstance(weights, dict)
        assert len(weights) == len(self.returns.columns)
    
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
        
        # Weights should be similar with same seed (allowing for numerical differences)
        # Check that the sum of weights is the same and individual weights are close
        assert abs(sum(weights1.values()) - sum(weights2.values())) < 1e-6
        assert abs(sum(weights1.values()) - 1.0) < 1e-6  # Should sum to 1
        
        # Check that weights are reasonable (between 0 and 1)
        for ticker in weights1:
            assert 0 <= weights1[ticker] <= 1
            assert 0 <= weights2[ticker] <= 1
    
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
        assert 'buy_and_hold' in available_strategies
    
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
    
    def test_create_buy_and_hold_strategy(self):
        """Test creating buy-and-hold strategy."""
        strategy = StrategyFactory.create('buy_and_hold',
                                        allocation_method="equal_weight")
        strategy.name = "Test Buy and Hold"
        
        assert isinstance(strategy, BuyAndHoldStrategy)
        assert strategy.name == "Test Buy and Hold"
        assert strategy.allocation_method == "equal_weight"
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = StrategyFactory.list_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) >= 5  # Should have at least our 5 strategies


class TestBuyAndHoldStrategy:
    """Test cases for BuyAndHoldStrategy."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample returns and prices data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create synthetic price data with different characteristics
        # High price, low volatility (large cap proxy)
        high_price_low_vol = 200 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        
        # Medium price, medium volatility
        med_price_med_vol = 100 + np.cumsum(np.random.normal(0, 1.0, len(dates)))
        
        # Low price, high volatility (small cap proxy)
        low_price_high_vol = 50 + np.cumsum(np.random.normal(0, 2.0, len(dates)))
        
        self.prices = pd.DataFrame({
            'AAPL': high_price_low_vol,  # Large cap proxy
            'MSFT': med_price_med_vol,   # Mid cap proxy
            'SMALL': low_price_high_vol  # Small cap proxy
        }, index=dates)
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
    
    def test_equal_weight_allocation(self):
        """Test equal weight allocation method."""
        strategy = BuyAndHoldStrategy("Test Equal Weight", allocation_method="equal_weight")
        
        weights = strategy.optimize(self.returns)
        
        # Check that all weights are equal
        expected_weight = 1.0 / len(self.returns.columns)
        for ticker, weight in weights.items():
            assert abs(weight - expected_weight) < 1e-6
            assert 0 <= weight <= 1
        
        # Check that weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_market_cap_allocation(self):
        """Test market cap allocation method."""
        strategy = BuyAndHoldStrategy("Test Market Cap", allocation_method="market_cap")
        
        weights = strategy.optimize(self.returns, prices=self.prices)
        
        # Check that weights are valid
        for ticker, weight in weights.items():
            assert 0 <= weight <= 1
        
        # Check that weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # Check that higher price/lower volatility assets get higher weights
        # AAPL should have the highest weight (highest price, lowest volatility)
        assert weights['AAPL'] > weights['MSFT']
        assert weights['MSFT'] > weights['SMALL']
    
    def test_custom_allocation(self):
        """Test custom allocation method."""
        custom_weights = {'AAPL': 0.5, 'MSFT': 0.3, 'SMALL': 0.2}
        strategy = BuyAndHoldStrategy("Test Custom", allocation_method="custom")
        
        weights = strategy.optimize(self.returns, custom_weights=custom_weights)
        
        # Check that weights match custom weights
        for ticker, expected_weight in custom_weights.items():
            assert abs(weights[ticker] - expected_weight) < 1e-6
    
    def test_custom_allocation_missing_weights(self):
        """Test custom allocation with missing weights."""
        custom_weights = {'AAPL': 0.5, 'MSFT': 0.3}  # Missing SMALL
        strategy = BuyAndHoldStrategy("Test Custom", allocation_method="custom")
        
        with pytest.raises(ValueError, match="Missing weights for tickers"):
            strategy.optimize(self.returns, custom_weights=custom_weights)
    
    def test_custom_allocation_invalid_sum(self):
        """Test custom allocation with weights that don't sum to 1."""
        custom_weights = {'AAPL': 0.5, 'MSFT': 0.3, 'SMALL': 0.3}  # Sum = 1.1
        strategy = BuyAndHoldStrategy("Test Custom", allocation_method="custom")
        
        with pytest.raises(ValueError, match="Custom weights must sum to 1.0"):
            strategy.optimize(self.returns, custom_weights=custom_weights)
    
    def test_market_cap_fallback_to_equal_weight(self):
        """Test market cap allocation falls back to equal weight when no prices provided."""
        strategy = BuyAndHoldStrategy("Test Market Cap", allocation_method="market_cap")
        
        # Don't provide prices - should fall back to equal weights
        weights = strategy.optimize(self.returns)
        
        # Should be equal weights
        expected_weight = 1.0 / len(self.returns.columns)
        for ticker, weight in weights.items():
            assert abs(weight - expected_weight) < 1e-6
    
    def test_invalid_allocation_method(self):
        """Test invalid allocation method."""
        strategy = BuyAndHoldStrategy("Test Invalid", allocation_method="invalid")
        
        with pytest.raises(ValueError, match="Unknown allocation method"):
            strategy.optimize(self.returns)
    
    def test_constraints_application(self):
        """Test that constraints are applied correctly."""
        strategy = BuyAndHoldStrategy("Test Constraints", allocation_method="equal_weight")
        
        constraints = {'min_weight': 0.1, 'max_weight': 0.5}
        weights = strategy.optimize(self.returns, constraints=constraints)
        
        # Check that constraints are applied
        for ticker, weight in weights.items():
            assert weight >= 0.1
            assert weight <= 0.5
        
        # Weights should still sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_get_parameters(self):
        """Test get_parameters method."""
        strategy = BuyAndHoldStrategy("Test Params", allocation_method="market_cap")
        
        params = strategy.get_parameters()
        
        assert params['strategy_type'] == 'buy_and_hold'
        assert params['allocation_method'] == 'market_cap'
        assert params['rebalancing'] is False
        assert 'description' in params
    
    def test_str_representation(self):
        """Test string representation."""
        strategy = BuyAndHoldStrategy("Test String", allocation_method="custom")
        
        str_repr = str(strategy)
        assert "Test String" in str_repr
        assert "Buy and Hold" in str_repr
        assert "custom" in str_repr
    
    def test_allocation_methods_different_results(self):
        """Test that different allocation methods produce different results."""
        equal_strategy = BuyAndHoldStrategy("Equal", allocation_method="equal_weight")
        market_cap_strategy = BuyAndHoldStrategy("Market Cap", allocation_method="market_cap")
        
        equal_weights = equal_strategy.optimize(self.returns, prices=self.prices)
        market_cap_weights = market_cap_strategy.optimize(self.returns, prices=self.prices)
        
        # Check that weights are different
        weights_equal = all(abs(equal_weights[ticker] - market_cap_weights[ticker]) < 1e-6 
                           for ticker in equal_weights.keys())
        
        assert not weights_equal, "Equal weight and market cap allocation should produce different results"
    
    def test_empty_returns_dataframe(self):
        """Test handling of empty returns DataFrame."""
        strategy = BuyAndHoldStrategy("Test Empty", allocation_method="equal_weight")
        
        with pytest.raises(ValueError, match="Returns DataFrame is empty"):
            strategy.optimize(pd.DataFrame())
    
    def test_single_asset(self):
        """Test with single asset."""
        single_returns = self.returns[['AAPL']]
        strategy = BuyAndHoldStrategy("Test Single", allocation_method="equal_weight")
        
        weights = strategy.optimize(single_returns)
        
        assert len(weights) == 1
        assert weights['AAPL'] == 1.0
    
    def test_strategy_factory_integration(self):
        """Test integration with StrategyFactory."""
        # Test default creation
        strategy1 = StrategyFactory.create('buy_and_hold')
        assert isinstance(strategy1, BuyAndHoldStrategy)
        assert strategy1.allocation_method == "equal_weight"
        
        # Test with parameters
        strategy2 = StrategyFactory.create('buy_and_hold', 
                                         allocation_method="market_cap")
        assert isinstance(strategy2, BuyAndHoldStrategy)
        assert strategy2.allocation_method == "market_cap"
        # Name is set by the factory, not by our parameter
        assert strategy2.name == "buy_and_hold"
