#!/usr/bin/env python3
"""Simple test script to verify basic functionality."""

import pandas as pd
import numpy as np
from datetime import datetime

from optfolio.data.loader import DataLoader
from optfolio.strategies.base import StrategyFactory
from optfolio.portfolio.base import Portfolio


def test_basic_functionality():
    """Test basic functionality with minimal data."""
    print("Testing basic functionality...")
    
    # Create simple test data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(42)
    
    # Create 3 simple assets
    data = {
        'AAPL': 100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))),
        'GOOGL': 200 + np.cumsum(np.random.normal(0.001, 0.025, len(dates))),
        'MSFT': 150 + np.cumsum(np.random.normal(0.0008, 0.018, len(dates)))
    }
    
    # Create temporary CSV files
    import tempfile
    import os
    from pathlib import Path
    
    temp_dir = tempfile.mkdtemp()
    prices_dir = Path(temp_dir) / "prices"
    prices_dir.mkdir()
    
    for ticker, prices in data.items():
        df = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        file_path = prices_dir / f"{ticker}.csv"
        df.to_csv(file_path, index=False)
    
    # Test data loading
    print("1. Testing data loading...")
    loader = DataLoader(prices_dir)
    prices = loader.load_prices()
    returns = loader.get_returns()
    
    print(f"   Loaded {len(prices.columns)} tickers")
    print(f"   Date range: {prices.index.min()} to {prices.index.max()}")
    
    # Test strategy creation
    print("\n2. Testing strategy creation...")
    equal_weight = StrategyFactory.create('equal_weight')
    print(f"   Created: {equal_weight}")
    
    # Test optimization
    print("\n3. Testing optimization...")
    weights = equal_weight.optimize(returns)
    print(f"   Weights: {weights}")
    
    # Test portfolio
    print("\n4. Testing portfolio...")
    portfolio = Portfolio(initial_capital=100000)
    
    # Get first date prices
    first_date = prices.index[0]
    current_prices = prices.loc[first_date].to_dict()
    
    # Set initial weights and rebalance
    portfolio.set_weights(weights)
    portfolio.rebalance(weights, current_prices)
    
    # Step through a few dates
    for i in range(10):
        date = prices.index[i]
        current_prices = prices.loc[date].to_dict()
        portfolio_value = portfolio.step(date, current_prices)
        print(f"   Date {date.strftime('%Y-%m-%d')}: ${portfolio_value:.2f}")
    
    # Test metrics
    print("\n5. Testing metrics...")
    portfolio_series = portfolio.get_portfolio_series()
    metrics = portfolio.get_performance_metrics()
    
    print(f"   Total return: {metrics.get('total_return', 0):.2%}")
    print(f"   Volatility: {metrics.get('volatility', 0):.2%}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n✅ All basic tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
