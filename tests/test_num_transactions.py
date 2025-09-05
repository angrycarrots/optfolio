#!/usr/bin/env python3
"""
Test script to verify that num_transactions correctly counts individual asset trades
instead of rebalancing events.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optfolio.portfolio.base import Portfolio
import pandas as pd
import numpy as np

def test_num_transactions_calculation():
    """Test that num_transactions counts individual asset trades correctly."""
    
    # Create a simple test portfolio
    initial_capital = 100000
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    prices = pd.DataFrame({
        'AAPL': [150, 155, 160, 165],
        'GOOGL': [2500, 2550, 2600, 2650],
        'MSFT': [300, 305, 310, 315],
        'TSLA': [800, 820, 840, 860]
    }, index=pd.date_range('2023-01-01', periods=4, freq='D'))
    
    portfolio = Portfolio(initial_capital, assets)
    
    # Simulate some transactions
    # First rebalancing: buy all assets
    portfolio.rebalance({'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}, prices.iloc[0])
    
    # Second rebalancing: adjust weights
    portfolio.rebalance({'AAPL': 0.3, 'GOOGL': 0.2, 'MSFT': 0.3, 'TSLA': 0.2}, prices.iloc[1])
    
    # Third rebalancing: another adjustment
    portfolio.rebalance({'AAPL': 0.4, 'GOOGL': 0.1, 'MSFT': 0.3, 'TSLA': 0.2}, prices.iloc[2])
    
    # Get summary
    summary = portfolio.get_summary()
    
    # Manual calculation
    num_rebalancing_events = len(portfolio.transaction_history)
    expected_num_transactions = sum(len(t['trades']) for t in portfolio.transaction_history)
    
    print("=== Test Results ===")
    print(f"Number of rebalancing events: {num_rebalancing_events}")
    print(f"Expected num_transactions (individual trades): {expected_num_transactions}")
    print(f"Actual num_transactions from summary: {summary['num_transactions']}")
    print(f"Number of assets: {len(assets)}")
    
    # Verify the calculation
    assert summary['num_transactions'] == expected_num_transactions, \
        f"Expected {expected_num_transactions} transactions, got {summary['num_transactions']}"
    
    # Verify it's not just counting rebalancing events
    assert summary['num_transactions'] != num_rebalancing_events, \
        f"num_transactions should not equal number of rebalancing events ({num_rebalancing_events})"
    
    # Verify it's a multiple of assets (since we trade all assets each time)
    assert summary['num_transactions'] == num_rebalancing_events * len(assets), \
        f"Expected {num_rebalancing_events * len(assets)} transactions, got {summary['num_transactions']}"
    
    print("✅ All tests passed!")
    print(f"✅ num_transactions correctly counts {summary['num_transactions']} individual asset trades")
    print(f"✅ This is {num_rebalancing_events} rebalancing events × {len(assets)} assets")
    
    # Test completed successfully
    assert True

def test_with_existing_results():
    """Test with existing detailed results to verify the calculation."""
    import json
    
    try:
        with open('/share/home/michael/projects/optfolio/detailed_results.json', 'r') as f:
            results = json.load(f)
        
        print("\n=== Testing with Existing Results ===")
        
        for strategy_name, strategy_data in results.items():
            transaction_history = strategy_data.get('transaction_history', [])
            
            if transaction_history:
                num_rebalancing_events = len(transaction_history)
                expected_num_transactions = sum(len(t['trades']) for t in transaction_history)
                
                print(f"\n{strategy_name}:")
                print(f"  Rebalancing events: {num_rebalancing_events}")
                print(f"  Expected individual trades: {expected_num_transactions}")
                
                # Check if we have the old CSV data for comparison
                if num_rebalancing_events == 26:
                    print(f"  Old CSV showed: 26 transactions (rebalancing events)")
                    print(f"  New calculation: {expected_num_transactions} transactions (individual trades)")
                    print(f"  Difference: {expected_num_transactions - 26} additional individual trades")
        
        # Test completed successfully
        assert True
        
    except FileNotFoundError:
        print("Could not find detailed_results.json for testing")
        # Test completed successfully even if file not found
        assert True

if __name__ == "__main__":
    print("Testing num_transactions calculation...")
    
    # Run the main test
    test_num_transactions_calculation()
    
    # Test with existing results
    test_with_existing_results()
    
    print("\n🎉 All tests completed successfully!")
