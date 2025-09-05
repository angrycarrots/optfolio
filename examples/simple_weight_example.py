#!/usr/bin/env python3
"""
Simple example showing how to create a DataFrame with weights for each symbol at each rebalance period.
"""

import pandas as pd
from optfolio.data.loader import DataLoader
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory

def main():
    """Simple example of extracting weight data."""
    
    # 1. Load data and run backtest
    print("Running backtest...")
    data_loader = DataLoader("price")
    
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    backtester.load_data(data_loader)
    
    # Create strategy
    strategy = StrategyFactory.create('black_litterman', 
                                    prior_method="market_cap", 
                                    view_method="momentum")
    
    # Run backtest
    result = backtester.run_backtest(
        strategy=strategy,
        rebalance_freq={"months": 3, "weeks": 1, "days": 1},
        start_date='2021-01-01',
        end_date='2023-12-31'
    )
    
    # 2. Extract weight DataFrame
    print("\nExtracting weight data...")
    weight_df = pd.DataFrame(result['weight_history'])
    
    # 3. Display information
    print(f"\nWeight DataFrame Info:")
    print(f"Shape: {weight_df.shape}")
    print(f"Rebalance dates: {len(weight_df)}")
    print(f"Symbols: {len(weight_df.columns)}")
    print(f"Date range: {weight_df.index.min()} to {weight_df.index.max()}")
    
    # 4. Show sample data
    print(f"\nSample weight data (first 3 rebalances, first 5 symbols):")
    print(weight_df.iloc[:3, :5].round(4))
    
    # 5. Basic analysis
    print(f"\nBasic weight analysis:")
    print(f"Average weight per symbol: {weight_df.mean().mean():.4f}")
    print(f"Maximum weight: {weight_df.max().max():.4f}")
    print(f"Minimum weight: {weight_df.min().min():.4f}")
    
    # 6. Save to CSV
    weight_df.to_csv("weights_simple.csv")
    print(f"\nWeight data saved to 'weights_simple.csv'")
    
    return weight_df

if __name__ == "__main__":
    weight_df = main()
