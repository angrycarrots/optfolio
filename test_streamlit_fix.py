#!/usr/bin/env python3
"""
Tests for the Streamlit app buy-and-hold fix.
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


class TestStreamlitFix(unittest.TestCase):
    """Test the Streamlit app buy-and-hold fix functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_dir = "data/price"
        # Use the same symbols as in streamlit_app.py
        self.TARGET_SYMBOLS = [
            'WSM', 'PAYX', 'BMI', 'BK', 'NDAQ', 'MSI', 'WMT', 'TJX', 'AIG', 'RJF', 
            'V', 'CTAS', 'TT', 'TRGP', 'JPM', 'GE', 'MCK', 'PH', 'LLY', 'COST', 
            'AVGO', 'NEE', 'AMAT', 'ADI', 'SHW', 'INTU', 'KLAC'
        ]
        
        # Load test data
        self.data_loader = DataLoader(self.data_dir)
        self.prices = self.data_loader.load_prices(tickers=self.TARGET_SYMBOLS)
        self.returns = self.data_loader.get_returns()
        self.returns = self.returns[self.prices.columns]
        
        # Set up backtester
        self.backtester = Backtester(
            initial_capital=100000.0,
            risk_free_rate=0.02,
            transaction_costs=0.001
        )
        self.backtester.load_data(self.data_loader, tickers=self.prices.columns.tolist())
    
    def create_strategies(self):
        """Create the same strategies as in streamlit_app.py."""
        strategies = [
            StrategyFactory.create('equal_weight'),
            StrategyFactory.create('buy_and_hold', allocation_method="equal_weight"),
            StrategyFactory.create('buy_and_hold', allocation_method="market_cap"),
            StrategyFactory.create('mean_variance', objective="sortino_ratio"),
            StrategyFactory.create('mean_variance', objective="sharpe_ratio"),
        ]
        
        strategy_names = [
            "Equal Weight", "Buy and Hold (Equal Weight)", "Buy and Hold (Market Cap)", 
            "Mean-Variance (Sortino)", "Mean-Variance (Sharpe)"
        ]
        
        for i, strategy in enumerate(strategies):
            strategy.name = strategy_names[i]
        
        return strategies
    
    def run_backtests_with_fix(self, rebalance_months):
        """Run backtests using the same logic as the fixed streamlit app."""
        strategies = self.create_strategies()
        
        # Define rebalancing schedule based on configuration (same as streamlit)
        if rebalance_months == 0:
            rebalance_freq = None
        else:
            rebalance_freq = {"months": rebalance_months, "weeks": 1, "days": 1}
        
        # Run backtests for each strategy individually to handle buy-and-hold correctly
        results = {}
        for strategy in strategies:
            # Buy-and-hold strategies should always use None rebalance_freq
            if 'buy and hold' in strategy.name.lower():
                strategy_rebalance_freq = None
            else:
                strategy_rebalance_freq = rebalance_freq
            
            # Run backtest for this strategy
            strategy_result = self.backtester.run_backtest(
                strategy=strategy,
                rebalance_freq=strategy_rebalance_freq,
                start_date='2023-01-01',
                end_date='2025-08-31'
            )
            results[strategy.name] = strategy_result
        
        return results
    
    def test_buy_and_hold_with_quarterly_rebalancing_setting(self):
        """Test that buy-and-hold strategies ignore the quarterly rebalancing setting."""
        # Test with quarterly rebalancing setting (rebalance_months = 3)
        results = self.run_backtests_with_fix(rebalance_months=3)
        
        # Check buy-and-hold strategies
        bh_equal_weight = results["Buy and Hold (Equal Weight)"]
        bh_market_cap = results["Buy and Hold (Market Cap)"]
        
        # Equal weight buy-and-hold should have 1 rebalance and 27 transactions
        self.assertEqual(bh_equal_weight['summary']['num_rebalances'], 1)
        self.assertEqual(bh_equal_weight['summary']['num_transactions'], len(self.TARGET_SYMBOLS))
        
        # Market cap buy-and-hold may have more rebalances due to market cap changes
        # But should have significantly fewer than regular rebalancing strategies
        self.assertGreaterEqual(bh_market_cap['summary']['num_rebalances'], 1)
        self.assertLess(bh_market_cap['summary']['num_rebalances'], 5)
        expected_transactions = bh_market_cap['summary']['num_rebalances'] * len(self.TARGET_SYMBOLS)
        self.assertEqual(bh_market_cap['summary']['num_transactions'], expected_transactions)
        
        # Check that other strategies use quarterly rebalancing
        equal_weight = results["Equal Weight"]
        self.assertGreater(equal_weight['summary']['num_rebalances'], 1)
        self.assertGreater(equal_weight['summary']['num_transactions'], len(self.TARGET_SYMBOLS))
    
    def test_buy_and_hold_with_buy_and_hold_setting(self):
        """Test that buy-and-hold strategies work correctly with buy-and-hold setting."""
        # Test with buy-and-hold setting (rebalance_months = 0)
        results = self.run_backtests_with_fix(rebalance_months=0)
        
        # Only check buy-and-hold strategies
        buy_and_hold_strategies = [name for name in results.keys() if 'buy and hold' in name.lower()]
        
        for strategy_name in buy_and_hold_strategies:
            result = results[strategy_name]
            summary = result['summary']
            if 'market cap' in strategy_name.lower():
                # Market cap buy-and-hold may have more rebalances
                self.assertGreaterEqual(summary['num_rebalances'], 1)
                self.assertLess(summary['num_rebalances'], 5)
                expected_transactions = summary['num_rebalances'] * len(self.TARGET_SYMBOLS)
                self.assertEqual(summary['num_transactions'], expected_transactions)
            else:
                # Other buy-and-hold strategies should have exactly 1 rebalance
                self.assertEqual(summary['num_rebalances'], 1)
                self.assertEqual(summary['num_transactions'], len(self.TARGET_SYMBOLS))
        
        # Check that we have the expected buy-and-hold strategies
        self.assertGreaterEqual(len(buy_and_hold_strategies), 2)
    
    def test_buy_and_hold_consistency_across_settings(self):
        """Test that buy-and-hold strategies are consistent regardless of rebalancing setting."""
        # Test with different rebalancing settings
        settings = [0, 1, 3, 6, 12]  # Different rebalancing frequencies
        
        for rebalance_months in settings:
            results = self.run_backtests_with_fix(rebalance_months)
            
            # Buy-and-hold strategies should always have the same results
            bh_equal_weight = results["Buy and Hold (Equal Weight)"]
            bh_market_cap = results["Buy and Hold (Market Cap)"]
            
            # Equal weight buy-and-hold should always have 1 rebalance
            self.assertEqual(bh_equal_weight['summary']['num_rebalances'], 1, 
                           f"Failed for rebalance_months={rebalance_months}")
            self.assertEqual(bh_equal_weight['summary']['num_transactions'], len(self.TARGET_SYMBOLS),
                           f"Failed for rebalance_months={rebalance_months}")
            
            # Market cap buy-and-hold may have more rebalances but should be consistent
            self.assertGreaterEqual(bh_market_cap['summary']['num_rebalances'], 1,
                           f"Failed for rebalance_months={rebalance_months}")
            self.assertLess(bh_market_cap['summary']['num_rebalances'], 5,
                           f"Failed for rebalance_months={rebalance_months}")
            expected_transactions = bh_market_cap['summary']['num_rebalances'] * len(self.TARGET_SYMBOLS)
            self.assertEqual(bh_market_cap['summary']['num_transactions'], expected_transactions,
                           f"Failed for rebalance_months={rebalance_months}")
    
    def test_strategy_name_matching(self):
        """Test that the strategy name matching logic works correctly."""
        test_cases = [
            ("Buy and Hold (Equal Weight)", True),
            ("Buy and Hold (Market Cap)", True),
            ("buy and hold (equal weight)", True),  # Case insensitive
            ("BUY AND HOLD (EQUAL WEIGHT)", True),  # Case insensitive
            ("Equal Weight", False),
            ("Mean-Variance (Sharpe)", False),
            ("Black-Litterman (Momentum)", False),
        ]
        
        for strategy_name, should_match in test_cases:
            matches = 'buy and hold' in strategy_name.lower()
            self.assertEqual(matches, should_match, 
                           f"Strategy name '{strategy_name}' matching failed")
    
    def test_transaction_history_structure(self):
        """Test that transaction history has the correct structure for buy-and-hold."""
        results = self.run_backtests_with_fix(rebalance_months=3)
        
        bh_result = results["Buy and Hold (Equal Weight)"]
        transaction_history = bh_result['transaction_history']
        
        # Should have exactly 1 transaction
        self.assertEqual(len(transaction_history), 1)
        
        # Check transaction structure
        transaction = transaction_history[0]
        self.assertIn('date', transaction)
        self.assertIn('trades', transaction)
        self.assertIn('transaction_cost', transaction)
        self.assertIn('portfolio_value', transaction)
        
        # Should have trades for all symbols
        self.assertEqual(len(transaction['trades']), len(self.TARGET_SYMBOLS))
        
        # All trades should be positive (buys)
        for symbol, shares in transaction['trades'].items():
            self.assertGreater(shares, 0, f"Symbol {symbol} should have positive shares")


if __name__ == '__main__':
    unittest.main()
