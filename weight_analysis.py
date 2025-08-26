#!/usr/bin/env python3
"""
Script to analyze portfolio weights across rebalance periods.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from optfolio.data.loader import DataLoader
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory

def create_weight_dataframe(backtest_result):
    """
    Create a DataFrame with weights for each symbol at each rebalance period.
    
    Args:
        backtest_result: Result dictionary from backtester.run_backtest()
        
    Returns:
        DataFrame with rebalance dates as index and symbols as columns
    """
    # Extract weight history from the backtest result
    weight_history = backtest_result['weight_history']
    
    # Convert to DataFrame
    weight_df = pd.DataFrame(weight_history)
    
    # The index should already be the rebalance dates
    # Let's ensure it's properly formatted
    weight_df.index.name = 'Rebalance_Date'
    
    return weight_df

def analyze_weight_changes(weight_df):
    """
    Analyze weight changes between rebalance periods.
    
    Args:
        weight_df: DataFrame with weights over time
        
    Returns:
        Dictionary with analysis results
    """
    # Calculate weight changes between periods
    weight_changes = weight_df.diff()
    
    # Calculate summary statistics
    analysis = {
        'total_rebalances': len(weight_df),
        'date_range': f"{weight_df.index.min().date()} to {weight_df.index.max().date()}",
        'symbols': list(weight_df.columns),
        'num_symbols': len(weight_df.columns),
        
        # Weight statistics
        'avg_weights': weight_df.mean(),
        'max_weights': weight_df.max(),
        'min_weights': weight_df.min(),
        'weight_volatility': weight_df.std(),
        
        # Change statistics
        'avg_changes': weight_changes.abs().mean(),
        'max_changes': weight_changes.abs().max(),
        'total_turnover': weight_changes.abs().sum().sum() / 2,  # Divide by 2 as each trade affects 2 positions
        
        # Concentration metrics
        'avg_concentration': (weight_df ** 2).sum(axis=1).mean(),  # Herfindahl index
        'max_concentration': (weight_df ** 2).sum(axis=1).max(),
    }
    
    return analysis, weight_changes

def print_weight_analysis(analysis, weight_df, weight_changes):
    """Print detailed weight analysis."""
    print("=" * 80)
    print("PORTFOLIO WEIGHT ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 Basic Information:")
    print(f"   Total Rebalances: {analysis['total_rebalances']}")
    print(f"   Date Range: {analysis['date_range']}")
    print(f"   Number of Symbols: {analysis['num_symbols']}")
    
    print(f"\n📈 Weight Statistics:")
    print(f"   Average Portfolio Turnover: {analysis['total_turnover']:.4f}")
    print(f"   Average Concentration (Herfindahl): {analysis['avg_concentration']:.4f}")
    print(f"   Maximum Concentration: {analysis['max_concentration']:.4f}")
    
    print(f"\n🏆 Top 5 Symbols by Average Weight:")
    top_symbols = analysis['avg_weights'].sort_values(ascending=False).head(5)
    for symbol, weight in top_symbols.items():
        print(f"   {symbol}: {weight:.4f}")
    
    print(f"\n📊 Top 5 Symbols by Weight Volatility:")
    top_vol = analysis['weight_volatility'].sort_values(ascending=False).head(5)
    for symbol, vol in top_vol.items():
        print(f"   {symbol}: {vol:.4f}")
    
    print(f"\n🔄 Top 5 Symbols by Average Weight Changes:")
    top_changes = analysis['avg_changes'].sort_values(ascending=False).head(5)
    for symbol, change in top_changes.items():
        print(f"   {symbol}: {change:.4f}")

def run_weight_analysis_example():
    """Run a complete weight analysis example."""
    print("Loading data and running backtest...")
    
    # Load data
    data_dir = "price"
    data_loader = DataLoader(data_dir)
    
    # Create backtester
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    
    # Load data into backtester
    backtester.load_data(data_loader)
    
    # Create strategy
    strategy = StrategyFactory.create('black_litterman', 
                                    prior_method="market_cap", 
                                    view_method="momentum")
    strategy.name = "Black-Litterman Analysis"
    
    # Run backtest
    rebalance_freq = {"months": 3, "weeks": 1, "days": 1}
    result = backtester.run_backtest(
        strategy=strategy,
        rebalance_freq=rebalance_freq,
        start_date='2021-01-01',
        end_date='2023-12-31'
    )
    
    # Create weight DataFrame
    weight_df = create_weight_dataframe(result)
    
    # Analyze weights
    analysis, weight_changes = analyze_weight_changes(weight_df)
    
    # Print analysis
    print_weight_analysis(analysis, weight_df, weight_changes)
    
    # Display the weight DataFrame
    print(f"\n📋 Weight DataFrame Preview:")
    print(f"Shape: {weight_df.shape}")
    print(f"Columns: {list(weight_df.columns)}")
    print(f"Index: {list(weight_df.index)}")
    
    print(f"\n📊 Weight DataFrame (first 5 rows, first 10 columns):")
    print(weight_df.iloc[:5, :10].round(4))
    
    # Save to CSV
    output_file = "portfolio_weights.csv"
    weight_df.to_csv(output_file)
    print(f"\n💾 Weight data saved to: {output_file}")
    
    return weight_df, analysis, weight_changes

def compare_strategies_weight_analysis():
    """Compare weight characteristics across different strategies."""
    print("\n" + "=" * 80)
    print("COMPARING WEIGHT CHARACTERISTICS ACROSS STRATEGIES")
    print("=" * 80)
    
    # Load data
    data_dir = "price"
    data_loader = DataLoader(data_dir)
    
    # Create backtester
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    backtester.load_data(data_loader)
    
    # Define strategies to test
    strategies = [
        ("Equal Weight", StrategyFactory.create('equal_weight')),
        ("Mean-Variance (Sortino)", StrategyFactory.create('mean_variance', objective="sortino_ratio")),
        ("Random Weight", StrategyFactory.create('random_weight', distribution="dirichlet", seed=42)),
        ("Black-Litterman", StrategyFactory.create('black_litterman', prior_method="market_cap", view_method="momentum"))
    ]
    
    rebalance_freq = {"months": 3, "weeks": 1, "days": 1}
    strategy_analyses = {}
    
    for strategy_name, strategy in strategies:
        print(f"\n🔄 Testing {strategy_name}...")
        strategy.name = strategy_name
        
        try:
            result = backtester.run_backtest(
                strategy=strategy,
                rebalance_freq=rebalance_freq,
                start_date='2021-01-01',
                end_date='2023-12-31'
            )
            
            weight_df = create_weight_dataframe(result)
            analysis, _ = analyze_weight_changes(weight_df)
            strategy_analyses[strategy_name] = analysis
            
            print(f"   ✅ Success - Turnover: {analysis['total_turnover']:.4f}, Concentration: {analysis['avg_concentration']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            strategy_analyses[strategy_name] = None
    
    # Compare strategies
    print(f"\n📊 Strategy Comparison:")
    print(f"{'Strategy':<25} {'Turnover':<12} {'Concentration':<15} {'Max Weight':<12} {'Min Weight':<12}")
    print("-" * 80)
    
    for strategy_name, analysis in strategy_analyses.items():
        if analysis:
            max_weight = analysis['max_weights'].max()
            min_weight = analysis['min_weights'].min()
            print(f"{strategy_name:<25} {analysis['total_turnover']:<12.4f} {analysis['avg_concentration']:<15.4f} {max_weight:<12.4f} {min_weight:<12.4f}")
        else:
            print(f"{strategy_name:<25} {'Failed':<12} {'Failed':<15} {'Failed':<12} {'Failed':<12}")

if __name__ == "__main__":
    print("=" * 80)
    print("PORTFOLIO WEIGHT ANALYSIS TOOL")
    print("=" * 80)
    
    # Run single strategy analysis
    weight_df, analysis, weight_changes = run_weight_analysis_example()
    
    # Run strategy comparison
    compare_strategies_weight_analysis()
    
    print(f"\n🎉 Analysis complete! Check 'portfolio_weights.csv' for detailed weight data.")
