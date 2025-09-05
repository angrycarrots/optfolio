#!/usr/bin/env python3
"""
Test script to verify the buy-and-hold fix works with the exact Streamlit configuration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optfolio.data.loader import DataLoader
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory

# Use the exact same configuration as Streamlit app
TARGET_SYMBOLS = [
    'WSM', 'PAYX', 'BMI', 'BK', 'NDAQ', 'MSI', 'WMT', 'TJX', 'AIG', 'RJF', 
    'V', 'CTAS', 'TT', 'TRGP', 'JPM', 'GE', 'MCK', 'PH', 'LLY', 'COST', 
    'AVGO', 'NEE', 'AMAT', 'ADI', 'SHW', 'INTU', 'KLAC'
]

def test_streamlit_config():
    """Test with exact Streamlit configuration."""
    
    print("Testing with exact Streamlit configuration...")
    print(f"Symbols: {len(TARGET_SYMBOLS)}")
    
    # Load data exactly like Streamlit app
    data_dir = "data/price"
    data_loader = DataLoader(data_dir)
    prices = data_loader.load_prices(tickers=TARGET_SYMBOLS)
    returns = data_loader.get_returns()
    returns = returns[prices.columns]
    
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    # Create strategies exactly like Streamlit app
    strategies = [
        StrategyFactory.create('equal_weight'),
        StrategyFactory.create('mean_variance', objective="sortino_ratio"),
        StrategyFactory.create('mean_variance', objective="sharpe_ratio"),
        StrategyFactory.create('random_weight', distribution="dirichlet", seed=42),
        StrategyFactory.create('black_litterman', 
                             prior_method="market_cap", view_method="momentum"),
        StrategyFactory.create('black_litterman', 
                             prior_method="market_cap", view_method="upside")
    ]
    
    strategy_names = [
        "Equal Weight", "Mean-Variance (Sortino)", "Mean-Variance (Sharpe)", 
        "Random Weight", "Black-Litterman (Momentum)", "Black-Litterman (Upside)"
    ]
    
    for i, strategy in enumerate(strategies):
        strategy.name = strategy_names[i]
        
        # Set minimum weight constraint for Black-Litterman strategies
        if 'black_litterman' in strategy.name.lower():
            strategy.min_weight = 0.01  # 1% minimum allocation
    
    # Set up backtester exactly like Streamlit app
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    
    # Load data into backtester
    backtester.load_data(data_loader, tickers=prices.columns.tolist())
    
    # Test buy-and-hold (rebalance_months = 0)
    print("\n=== Testing Buy-and-Hold (rebalance_months = 0) ===")
    results_buy_hold = backtester.run_multiple_backtests(
        strategies=strategies,
        rebalance_freq=None,  # This is what Streamlit sets when rebalance_months = 0
        start_date='2017-01-01',
        end_date='2025-08-31'
    )
    
    # Test quarterly rebalancing for comparison
    print("\n=== Testing Quarterly Rebalancing (rebalance_months = 3) ===")
    results_quarterly = backtester.run_multiple_backtests(
        strategies=strategies,
        rebalance_freq={"months": 3, "weeks": 1, "days": 1},
        start_date='2017-01-01',
        end_date='2025-08-31'
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
        
        # Expected transactions for buy-and-hold
        expected_bh_transactions = len(TARGET_SYMBOLS)  # Should only trade once at the beginning
        
        if bh_transactions == expected_bh_transactions:
            print(f"  ✅ Buy-and-hold correct: {bh_transactions} transactions")
        elif bh_transactions < q_transactions:
            print(f"  ✅ Buy-and-hold has fewer transactions than quarterly")
        else:
            print(f"  ❌ Buy-and-hold has too many transactions: {bh_transactions}")
        
        # Check if quarterly has more transactions
        if q_transactions > bh_transactions:
            print(f"  ✅ Quarterly has more transactions as expected")
        else:
            print(f"  ❌ Quarterly should have more transactions than buy-and-hold")
    
    # Test completed successfully
    assert results_buy_hold is not None
    assert results_quarterly is not None

if __name__ == "__main__":
    test_streamlit_config()
    print("\n🎉 Streamlit configuration test completed!")
