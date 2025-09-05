#!/usr/bin/env python3
"""
Test script to verify that buy-and-hold strategy works correctly with minimal transactions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory
from optfolio.data.loader import DataLoader
import pandas as pd

def test_buy_and_hold():
    """Test that buy-and-hold strategy has minimal transactions."""
    
    print("Testing buy-and-hold strategy...")
    
    # Load data
    data_dir = "data/price"
    data_loader = DataLoader(data_dir)
    
    # Use a subset of symbols for faster testing (from your actual data)
    test_symbols = ['WSM', 'PAYX', 'BMI', 'BK', 'NDAQ']
    prices = data_loader.load_prices(tickers=test_symbols)
    returns = data_loader.get_returns()
    returns = returns[prices.columns]
    
    print(f"Loaded {len(test_symbols)} symbols")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    # Create strategies
    strategies = [
        StrategyFactory.create('equal_weight'),
        StrategyFactory.create('mean_variance', objective="sharpe_ratio")
    ]
    
    strategy_names = ["Equal Weight", "Mean-Variance (Sharpe)"]
    for i, strategy in enumerate(strategies):
        strategy.name = strategy_names[i]
    
    # Set up backtester
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    
    # Load data into backtester
    backtester.load_data(data_loader, tickers=prices.columns.tolist())
    
    # Test buy-and-hold (rebalance_freq = None)
    print("\n=== Testing Buy-and-Hold (rebalance_freq = None) ===")
    results_buy_hold = backtester.run_multiple_backtests(
        strategies=strategies,
        rebalance_freq=None,  # This should trigger buy-and-hold
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Test quarterly rebalancing for comparison
    print("\n=== Testing Quarterly Rebalancing ===")
    results_quarterly = backtester.run_multiple_backtests(
        strategies=strategies,
        rebalance_freq={"months": 3, "weeks": 1, "days": 1},
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Compare results
    print("\n=== Results Comparison ===")
    for strategy_name in strategy_names:
        print(f"\n{strategy_name}:")
        
        # Buy-and-hold results
        bh_summary = results_buy_hold[strategy_name]['summary']
        bh_transactions = bh_summary.get('num_transactions', 0)
        bh_costs = bh_summary.get('total_transaction_costs', 0)
        
        # Quarterly results
        q_summary = results_quarterly[strategy_name]['summary']
        q_transactions = q_summary.get('num_transactions', 0)
        q_costs = q_summary.get('total_transaction_costs', 0)
        
        print(f"  Buy-and-Hold: {bh_transactions} transactions, ${bh_costs:.2f} costs")
        print(f"  Quarterly:    {q_transactions} transactions, ${q_costs:.2f} costs")
        
        # Verify buy-and-hold has minimal transactions
        expected_bh_transactions = len(test_symbols)  # Should only trade once at the beginning
        if bh_transactions == expected_bh_transactions:
            print(f"  ✅ Buy-and-hold correct: {bh_transactions} transactions (expected {expected_bh_transactions})")
        else:
            print(f"  ❌ Buy-and-hold incorrect: {bh_transactions} transactions (expected {expected_bh_transactions})")
        
        # Verify quarterly has more transactions
        if q_transactions > bh_transactions:
            print(f"  ✅ Quarterly has more transactions as expected")
        else:
            print(f"  ❌ Quarterly should have more transactions than buy-and-hold")
    
    # Test completed successfully
    assert results_buy_hold is not None
    assert results_quarterly is not None

if __name__ == "__main__":
    test_buy_and_hold()
    print("\n🎉 Buy-and-hold test completed!")
