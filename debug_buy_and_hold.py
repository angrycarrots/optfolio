#!/usr/bin/env python3
"""
Debug script to understand what's happening with buy-and-hold transactions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory
from optfolio.data.loader import DataLoader
from optfolio.portfolio.base import Portfolio
import pandas as pd

def debug_buy_and_hold():
    """Debug buy-and-hold strategy to understand transaction behavior."""
    
    print("Debugging buy-and-hold strategy...")
    
    # Load data
    data_dir = "data/price"
    data_loader = DataLoader(data_dir)
    
    # Use a small subset for debugging
    test_symbols = ['WSM', 'PAYX']
    prices = data_loader.load_prices(tickers=test_symbols)
    returns = data_loader.get_returns()
    returns = returns[prices.columns]
    
    print(f"Loaded {len(test_symbols)} symbols")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    # Create a simple equal weight strategy
    strategy = StrategyFactory.create('equal_weight')
    strategy.name = "Equal Weight"
    
    # Set up backtester
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    
    # Load data into backtester
    backtester.load_data(data_loader, tickers=prices.columns.tolist())
    
    # Test buy-and-hold manually
    print("\n=== Manual Buy-and-Hold Test ===")
    
    # Create portfolio manually
    portfolio = Portfolio(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    
    print(f"Initial capital: {portfolio.current_capital}")
    print(f"Initial positions: {portfolio.positions}")
    print(f"Initial portfolio value: {portfolio.calculate_portfolio_value()}")
    
    # Get first date and prices
    first_date = prices.index[0]
    first_prices = prices.loc[first_date].to_dict()
    print(f"\nFirst date: {first_date}")
    print(f"First prices: {first_prices}")
    
    # Get historical returns for optimization
    historical_returns = returns[returns.index < first_date]
    print(f"Historical returns shape: {historical_returns.shape}")
    
    if len(historical_returns) > 0:
        # Optimize initial weights
        initial_weights = strategy.optimize(historical_returns)
        print(f"Initial weights: {initial_weights}")
        
        # Set initial weights
        portfolio.set_weights(initial_weights)
        print(f"Weights after setting: {portfolio.weights}")
        
        # Initial rebalance
        print(f"\nBefore rebalance:")
        print(f"  Positions: {portfolio.positions}")
        print(f"  Portfolio value: {portfolio.calculate_portfolio_value()}")
        
        trades = portfolio.rebalance(initial_weights, first_prices)
        print(f"\nAfter rebalance:")
        print(f"  Trades executed: {trades}")
        print(f"  Positions: {portfolio.positions}")
        print(f"  Portfolio value: {portfolio.calculate_portfolio_value()}")
        print(f"  Transaction history length: {len(portfolio.transaction_history)}")
        
        if portfolio.transaction_history:
            print(f"  Last transaction: {portfolio.transaction_history[-1]}")
        
        # Get summary
        summary = portfolio.get_summary()
        print(f"\nSummary:")
        print(f"  num_transactions: {summary.get('num_transactions', 0)}")
        print(f"  total_transaction_costs: {summary.get('total_transaction_costs', 0)}")
    else:
        print("No historical returns available for optimization")
    
    # Now test with the backtester
    print("\n=== Backtester Buy-and-Hold Test ===")
    results = backtester.run_multiple_backtests(
        strategies=[strategy],
        rebalance_freq=None,  # Buy-and-hold
        start_date='2023-01-01',
        end_date='2023-01-31'  # Short period for debugging
    )
    
    if results:
        result = results["Equal Weight"]
        summary = result['summary']
        print(f"Backtester results:")
        print(f"  num_transactions: {summary.get('num_transactions', 0)}")
        print(f"  total_transaction_costs: {summary.get('total_transaction_costs', 0)}")
        
        if 'transaction_history' in result:
            print(f"  transaction_history length: {len(result['transaction_history'])}")
            if result['transaction_history']:
                print(f"  First transaction: {result['transaction_history'][0]}")
                print(f"  Last transaction: {result['transaction_history'][-1]}")

if __name__ == "__main__":
    debug_buy_and_hold()
