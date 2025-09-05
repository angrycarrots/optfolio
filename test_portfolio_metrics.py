#!/usr/bin/env python3
"""
Tests for Portfolio class metrics, specifically the new num_rebalances metric.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from optfolio.portfolio.base import Portfolio


class TestPortfolioMetrics(unittest.TestCase):
    """Test Portfolio class metrics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_capital = 100000.0
        self.assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            risk_free_rate=0.02,
            transaction_costs=0.001
        )
        
        # Create test prices
        self.prices = pd.DataFrame({
            'AAPL': [150, 155, 160, 165],
            'GOOGL': [2500, 2550, 2600, 2650],
            'MSFT': [300, 305, 310, 315],
            'TSLA': [800, 820, 840, 860]
        }, index=pd.date_range('2023-01-01', periods=4, freq='D'))
    
    def test_initial_state(self):
        """Test initial portfolio state."""
        summary = self.portfolio.get_summary()
        
        self.assertEqual(summary['num_rebalances'], 0)
        self.assertEqual(summary['num_transactions'], 0)
        self.assertEqual(summary['total_transaction_costs'], 0.0)
        self.assertEqual(summary['initial_capital'], self.initial_capital)
    
    def test_single_rebalance(self):
        """Test metrics after a single rebalancing event."""
        # Single rebalance with all assets
        target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        current_prices = self.prices.iloc[0].to_dict()
        
        self.portfolio.rebalance(target_weights, current_prices)
        
        summary = self.portfolio.get_summary()
        
        # Should have 1 rebalancing event
        self.assertEqual(summary['num_rebalances'], 1)
        # Should have 4 transactions (one per asset)
        self.assertEqual(summary['num_transactions'], 4)
        # Should have transaction costs
        self.assertGreater(summary['total_transaction_costs'], 0)
    
    def test_multiple_rebalances(self):
        """Test metrics after multiple rebalancing events."""
        # First rebalance
        target_weights_1 = {'AAPL': 0.3, 'GOOGL': 0.2, 'MSFT': 0.3, 'TSLA': 0.2}
        self.portfolio.rebalance(target_weights_1, self.prices.iloc[0].to_dict())
        
        # Second rebalance
        target_weights_2 = {'AAPL': 0.4, 'GOOGL': 0.1, 'MSFT': 0.3, 'TSLA': 0.2}
        self.portfolio.rebalance(target_weights_2, self.prices.iloc[1].to_dict())
        
        # Third rebalance
        target_weights_3 = {'AAPL': 0.5, 'GOOGL': 0.1, 'MSFT': 0.2, 'TSLA': 0.2}
        self.portfolio.rebalance(target_weights_3, self.prices.iloc[2].to_dict())
        
        summary = self.portfolio.get_summary()
        
        # Should have 3 rebalancing events
        self.assertEqual(summary['num_rebalances'], 3)
        # Should have 12 transactions (3 rebalances × 4 assets)
        self.assertEqual(summary['num_transactions'], 12)
        # Should have accumulated transaction costs
        self.assertGreater(summary['total_transaction_costs'], 0)
    
    def test_buy_and_hold_scenario(self):
        """Test metrics for a buy-and-hold scenario (single rebalance)."""
        # Single rebalance at the beginning (buy-and-hold)
        target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        self.portfolio.rebalance(target_weights, self.prices.iloc[0].to_dict())
        
        # Step through time without rebalancing
        for i in range(1, len(self.prices)):
            self.portfolio.step(self.prices.index[i], self.prices.iloc[i].to_dict())
        
        summary = self.portfolio.get_summary()
        
        # Should still have only 1 rebalancing event
        self.assertEqual(summary['num_rebalances'], 1)
        # Should still have only 4 transactions
        self.assertEqual(summary['num_transactions'], 4)
    
    def test_rebalance_with_no_trades(self):
        """Test rebalancing when no trades are needed."""
        # Initial rebalance
        target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        self.portfolio.rebalance(target_weights, self.prices.iloc[0].to_dict())
        
        initial_summary = self.portfolio.get_summary()
        
        # Rebalance to the same weights (should result in no trades)
        self.portfolio.rebalance(target_weights, self.prices.iloc[1].to_dict())
        
        final_summary = self.portfolio.get_summary()
        
        # Should have 2 rebalancing events
        self.assertEqual(final_summary['num_rebalances'], 2)
        # Should have 8 transactions (2 rebalances × 4 assets, even if no actual trades)
        self.assertEqual(final_summary['num_transactions'], 8)
        # Transaction costs should be higher (2 rebalances)
        self.assertGreaterEqual(final_summary['total_transaction_costs'], 
                               initial_summary['total_transaction_costs'])
    
    def test_partial_rebalance(self):
        """Test rebalancing with only some assets."""
        # Initial rebalance with all assets
        target_weights_1 = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        self.portfolio.rebalance(target_weights_1, self.prices.iloc[0].to_dict())
        
        # Rebalance with only 2 assets (others stay the same)
        target_weights_2 = {'AAPL': 0.4, 'GOOGL': 0.1, 'MSFT': 0.25, 'TSLA': 0.25}
        self.portfolio.rebalance(target_weights_2, self.prices.iloc[1].to_dict())
        
        summary = self.portfolio.get_summary()
        
        # Should have 2 rebalancing events
        self.assertEqual(summary['num_rebalances'], 2)
        # Should have 8 transactions (2 rebalances × 4 assets)
        self.assertEqual(summary['num_transactions'], 8)


if __name__ == '__main__':
    unittest.main()
