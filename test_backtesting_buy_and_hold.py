#!/usr/bin/env python3
"""
Tests for backtesting engine buy-and-hold strategy behavior.
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


class TestBacktestingBuyAndHold(unittest.TestCase):
    """Test backtesting engine buy-and-hold functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_dir = "data/price"
        self.test_symbols = ['WSM', 'PAYX', 'BMI', 'BK', 'NDAQ']  # 5 symbols for faster testing
        
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
    
    def test_buy_and_hold_with_none_rebalance_freq(self):
        """Test buy-and-hold strategy with rebalance_freq=None."""
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold (Equal Weight)"
        
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq=None,  # Buy and hold
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        summary = results['summary']
        
        # Should have exactly 1 rebalancing event
        self.assertEqual(summary['num_rebalances'], 1)
        # Should have exactly 5 transactions (one per symbol)
        self.assertEqual(summary['num_transactions'], len(self.test_symbols))
        # Should have transaction history with 1 entry
        self.assertEqual(len(results['transaction_history']), 1)
        
        # Check that all symbols were traded in the first transaction
        first_transaction = results['transaction_history'][0]
        traded_symbols = set(first_transaction['trades'].keys())
        expected_symbols = set(self.test_symbols)
        self.assertEqual(traded_symbols, expected_symbols)
    
    def test_buy_and_hold_with_zero_rebalance_freq(self):
        """Test buy-and-hold strategy with rebalance_freq={"months": 0, "weeks": 0, "days": 0}."""
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold (Equal Weight)"
        
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq={"months": 0, "weeks": 0, "days": 0},  # Buy and hold
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        summary = results['summary']
        
        # Should have exactly 1 rebalancing event
        self.assertEqual(summary['num_rebalances'], 1)
        # Should have exactly 5 transactions (one per symbol)
        self.assertEqual(summary['num_transactions'], len(self.test_symbols))
    
    def test_equal_weight_with_quarterly_rebalancing(self):
        """Test equal weight strategy with quarterly rebalancing for comparison."""
        strategy = StrategyFactory.create('equal_weight')
        strategy.name = "Equal Weight"
        
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq={"months": 3},  # Quarterly rebalancing
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        summary = results['summary']
        
        # Should have multiple rebalancing events (quarterly)
        self.assertGreater(summary['num_rebalances'], 1)
        # Should have multiple transactions (rebalances × symbols)
        expected_transactions = summary['num_rebalances'] * len(self.test_symbols)
        self.assertEqual(summary['num_transactions'], expected_transactions)
    
    def test_buy_and_hold_vs_equal_weight_comparison(self):
        """Test that buy-and-hold has fewer transactions than equal weight."""
        # Buy and hold strategy
        bh_strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        bh_strategy.name = "Buy and Hold"
        
        bh_results = self.backtester.run_backtest(
            strategy=bh_strategy,
            rebalance_freq=None,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Equal weight strategy
        ew_strategy = StrategyFactory.create('equal_weight')
        ew_strategy.name = "Equal Weight"
        
        ew_results = self.backtester.run_backtest(
            strategy=ew_strategy,
            rebalance_freq={"months": 3},  # Quarterly
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        bh_summary = bh_results['summary']
        ew_summary = ew_results['summary']
        
        # Buy and hold should have fewer rebalances
        self.assertLess(bh_summary['num_rebalances'], ew_summary['num_rebalances'])
        # Buy and hold should have fewer transactions
        self.assertLess(bh_summary['num_transactions'], ew_summary['num_transactions'])
        # Buy and hold should have lower transaction costs
        self.assertLess(bh_summary['total_transaction_costs'], ew_summary['total_transaction_costs'])
    
    def test_buy_and_hold_market_cap_allocation(self):
        """Test buy-and-hold strategy with market cap allocation."""
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="market_cap")
        strategy.name = "Buy and Hold (Market Cap)"
        
        results = self.backtester.run_backtest(
            strategy=strategy,
            rebalance_freq=None,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        summary = results['summary']
        
        # Market cap buy-and-hold may have more than 1 rebalance due to market cap changes
        # But should have significantly fewer than a regular rebalancing strategy
        self.assertGreaterEqual(summary['num_rebalances'], 1)
        self.assertLess(summary['num_rebalances'], 5)  # Much less than quarterly rebalancing
        # Should have transactions equal to rebalances × symbols
        expected_transactions = summary['num_rebalances'] * len(self.test_symbols)
        self.assertEqual(summary['num_transactions'], expected_transactions)
    
    def test_multiple_strategies_with_different_rebalance_freqs(self):
        """Test running multiple strategies with different rebalancing frequencies."""
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
                rebalance_freq = None  # Buy and hold
            else:
                rebalance_freq = {"months": 3}  # Quarterly
            
            strategy_result = self.backtester.run_backtest(
                strategy=strategy,
                rebalance_freq=rebalance_freq,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            results[strategy.name] = strategy_result
        
        # Verify buy and hold has fewer transactions
        bh_summary = results["Buy and Hold"]['summary']
        ew_summary = results["Equal Weight"]['summary']
        
        self.assertEqual(bh_summary['num_rebalances'], 1)
        self.assertGreater(ew_summary['num_rebalances'], 1)
        self.assertLess(bh_summary['num_transactions'], ew_summary['num_transactions'])


if __name__ == '__main__':
    unittest.main()
