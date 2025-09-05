#!/usr/bin/env python3
"""
Integration tests to verify end-to-end functionality of the buy-and-hold fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory
from optfolio.data.loader import DataLoader


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete buy-and-hold fix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_dir = "data/price"
        # Use a subset of symbols for faster testing
        self.test_symbols = ['WSM', 'PAYX', 'BMI', 'BK', 'NDAQ', 'MSI', 'WMT', 'TJX']
        
        # Load test data
        self.data_loader = DataLoader(self.data_dir)
        self.prices = self.data_loader.load_prices(tickers=self.test_symbols)
        self.returns = self.data_loader.get_returns()
        self.returns = self.returns[self.prices.columns]
        
        # Set up backtester
        self.backtester = Backtester(
            initial_capital=100000.0,
            risk_free_rate=0.02,
            transaction_costs=0.001
        )
        self.backtester.load_data(self.data_loader, tickers=self.prices.columns.tolist())
    
    def test_complete_buy_and_hold_workflow(self):
        """Test the complete buy-and-hold workflow from strategy creation to results."""
        # Create buy-and-hold strategy
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold (Equal Weight)"
        
        # Run backtest with buy-and-hold configuration
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq=None,  # Buy and hold
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Verify all expected components are present
        self.assertIn('summary', results)
        self.assertIn('portfolio_values', results)
        self.assertIn('weight_history', results)
        self.assertIn('performance_metrics', results)
        self.assertIn('transaction_history', results)
        
        # Verify buy-and-hold behavior
        summary = results['summary']
        self.assertEqual(summary['num_rebalances'], 1)
        self.assertEqual(summary['num_transactions'], len(self.test_symbols))
        
        # Verify transaction history
        transaction_history = results['transaction_history']
        self.assertEqual(len(transaction_history), 1)
        
        # Verify portfolio values are tracked
        portfolio_values = results['portfolio_values']
        self.assertGreater(len(portfolio_values), 0)
        
        # Verify weight history
        weight_history = results['weight_history']
        self.assertIsNotNone(weight_history)
    
    def test_metrics_consistency(self):
        """Test that the new metrics are consistent across different scenarios."""
        strategies = [
            StrategyFactory.create('equal_weight'),
            StrategyFactory.create('buy_and_hold', allocation_method="equal_weight"),
        ]
        
        strategy_names = ["Equal Weight", "Buy and Hold"]
        for i, strategy in enumerate(strategies):
            strategy.name = strategy_names[i]
        
        # Test with quarterly rebalancing
        results = {}
        for strategy in strategies:
            if 'buy and hold' in strategy.name.lower():
                rebalance_freq = None
            else:
                rebalance_freq = {"months": 3}
            
            strategy_result = self.backtester.run_backtest(
                strategy=strategy,
                rebalance_freq=rebalance_freq,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            results[strategy.name] = strategy_result
        
        # Verify metrics consistency
        for strategy_name, result in results.items():
            summary = result['summary']
            
            # num_rebalances should be a positive integer
            self.assertIsInstance(summary['num_rebalances'], int)
            self.assertGreaterEqual(summary['num_rebalances'], 1)
            
            # num_transactions should be a positive integer
            self.assertIsInstance(summary['num_transactions'], int)
            self.assertGreaterEqual(summary['num_transactions'], 1)
            
            # num_transactions should be a multiple of num_rebalances and number of symbols
            # (assuming all symbols are traded in each rebalance)
            self.assertEqual(summary['num_transactions'] % len(self.test_symbols), 0)
            
            # Transaction costs should be non-negative
            self.assertGreaterEqual(summary['total_transaction_costs'], 0)
    
    def test_performance_metrics_calculation(self):
        """Test that performance metrics are calculated correctly."""
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold"
        
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq=None,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        performance_metrics = results['performance_metrics']
        
        # Verify key performance metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, performance_metrics)
            self.assertIsInstance(performance_metrics[metric], (int, float))
    
    def test_portfolio_value_tracking(self):
        """Test that portfolio values are tracked correctly over time."""
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold"
        
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq=None,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        portfolio_values = results['portfolio_values']
        
        # Should have values for the entire date range
        self.assertGreater(len(portfolio_values), 0)
        
        # All values should be positive
        for value in portfolio_values:
            self.assertGreater(value, 0)
        
        # Values should be in chronological order
        dates = portfolio_values.index
        self.assertTrue(dates.is_monotonic_increasing)
    
    def test_weight_history_tracking(self):
        """Test that weight history is tracked correctly."""
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold"
        
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq=None,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        weight_history = results['weight_history']
        
        # Should have weight history
        self.assertIsNotNone(weight_history)
        
        if hasattr(weight_history, 'columns'):
            # Should have weights for all symbols
            self.assertEqual(len(weight_history.columns), len(self.test_symbols))
            
            # All weights should be between 0 and 1
            for col in weight_history.columns:
                weights = weight_history[col]
                self.assertTrue((weights >= 0).all())
                self.assertTrue((weights <= 1).all())
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold"
        
        # Test with invalid date range
        with self.assertRaises(Exception):
            self.backtester.run_backtest(
                strategy=strategy,
                rebalance_freq=None,
                start_date='2025-01-01',  # Future date
                end_date='2023-12-31'     # Past date
            )
    
    def test_multiple_strategies_comparison(self):
        """Test comparing multiple strategies with different behaviors."""
        strategies = [
            StrategyFactory.create('equal_weight'),
            StrategyFactory.create('buy_and_hold', allocation_method="equal_weight"),
            StrategyFactory.create('mean_variance', objective="sharpe_ratio"),
        ]
        
        strategy_names = ["Equal Weight", "Buy and Hold", "Mean-Variance"]
        for i, strategy in enumerate(strategies):
            strategy.name = strategy_names[i]
        
        # Run all strategies with quarterly rebalancing
        results = {}
        for strategy in strategies:
            if 'buy and hold' in strategy.name.lower():
                rebalance_freq = None
            else:
                rebalance_freq = {"months": 3}
            
            strategy_result = self.backtester.run_backtest(
                strategy=strategy,
                rebalance_freq=rebalance_freq,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            results[strategy.name] = strategy_result
        
        # Verify all strategies completed successfully
        self.assertEqual(len(results), 3)
        
        # Verify buy-and-hold has the expected behavior
        bh_result = results["Buy and Hold"]
        self.assertEqual(bh_result['summary']['num_rebalances'], 1)
        self.assertEqual(bh_result['summary']['num_transactions'], len(self.test_symbols))
        
        # Verify other strategies have different behavior
        ew_result = results["Equal Weight"]
        mv_result = results["Mean-Variance"]
        
        self.assertGreater(ew_result['summary']['num_rebalances'], 1)
        self.assertGreater(mv_result['summary']['num_rebalances'], 1)


if __name__ == '__main__':
    unittest.main()
