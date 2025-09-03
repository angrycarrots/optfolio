#!/usr/bin/env python3
"""
Example script demonstrating the Black-Litterman strategy with upside views.

This script shows how to:
1. Load data from CSV files
2. Create and configure Black-Litterman strategy with upside views
3. Run backtest with quarterly rebalancing
4. Analyze performance and generate reports
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from optfolio.data.loader import DataLoader
from optfolio.data.validator import DataValidator
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory


def run_black_litterman_upside_example():
    """Run the Black-Litterman with upside views example."""
    print("=" * 70)
    print("BLACK-LITTERMAN STRATEGY WITH UPSIDE VIEWS EXAMPLE")
    print("=" * 70)
    
    # Step 1: Set up data directory and load data
    print("\n1. Loading and validating data...")
    data_dir = "price"
    
    data_loader = DataLoader(data_dir)
    
    # Use a subset of symbols that likely have analyst coverage
    # symbols = [
    #     'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
    #     'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'DIS', 'PYPL', 'INTC', 'V', 'MA'
    # ]
    symbols = ['WSM', 'PAYX', 'BMI', 'BK', 'NDAQ', 'MSI', 'WMT', 'TJX', 'AIG', 
                'RJF', 'V', 'CTAS', 'TT', 'TRGP', 'JPM', 'GE', 'MCK', 'PH', 'LLY', 
                'COST', 'AVGO', 'NEE', 'AMAT', 'ADI', 'SHW', 'INTU', 'KLAC']
    # Load price data for selected symbols
    try:
        prices = data_loader.load_prices(tickers=symbols)
        returns = data_loader.get_returns()
        print(f"   Loaded {len(prices.columns)} tickers")
        print(f"   Date range: {prices.index.min()} to {prices.index.max()}")
        print(f"   Total observations: {len(prices)}")
    except Exception as e:
        print(f"   Warning: Could not load all symbols, using available data: {e}")
        # Fallback to available data
        prices = data_loader.load_prices()
        returns = data_loader.get_returns()
        symbols = list(prices.columns)
        print(f"   Using {len(symbols)} available symbols")
    
    # Validate data
    validator = DataValidator()
    price_issues = validator.validate_price_data(prices)
    returns_issues = validator.validate_returns_data(returns)
    
    print(f"   Data quality score: {validator.get_data_quality_score(prices):.3f}")
    
    if price_issues['warnings']:
        print(f"   Warnings: {len(price_issues['warnings'])}")
    if price_issues['errors']:
        print(f"   Errors: {len(price_issues['errors'])}")
    
    # Step 2: Create Black-Litterman strategy with upside views
    print("\n2. Creating Black-Litterman strategy with upside views...")
    
    # Create strategy with upside views
    strategy = StrategyFactory.create('black_litterman', 
                                    prior_method="market_cap",  # Use market cap weighted priors
                                    view_method="upside")       # Use analyst upside views
    
    # Set minimum weight constraint (1% minimum allocation per asset)
    strategy.min_weight = 0.01
    
    strategy.name = "Black-Litterman (Upside Views)"
    print(f"   Created: {strategy}")
    print(f"   Prior method: {strategy.prior_method}")
    print(f"   View method: {strategy.view_method}")
    
    # Step 3: Set up backtesting
    print("\n3. Setting up backtesting...")
    
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    
    # Load data into backtester
    backtester.load_data(data_loader, tickers=symbols)
    
    # Step 4: Run backtest with quarterly rebalancing
    print("\n4. Running backtest with quarterly rebalancing...")
    
    # Define quarterly rebalancing (every 3 months)
    rebalance_freq = {"months": 3}
    
    # Run backtest
    result = backtester.run_backtest(
        strategy=strategy,
        rebalance_freq=rebalance_freq,
        start_date='2017-01-01',  # Start from 2020 to have enough history
        end_date='2025-06-30'
    )
    
    print(f"   Backtest completed successfully")
    
    # Step 5: Analyze results
    print("\n5. Analyzing results...")
    
    metrics = result['performance_metrics']
    summary = result['summary']
    
    print(f"\nPerformance Summary:")
    print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Number of Transactions: {summary.get('num_transactions', 0)}")
    print(f"  Total Transaction Costs: ${summary.get('total_transaction_costs', 0):.2f}")
    
    # Step 6: Analyze asset allocation over time
    print("\n6. Analyzing asset allocation over time...")
    
    if 'weight_history' in result:
        weight_df = pd.DataFrame(result['weight_history'])
        print(f"   Weight history shape: {weight_df.shape}")
        print(f"   Rebalancing dates: {len(weight_df)}")
        
        # Show final weights
        if 'last_weights' in result:
            final_weights = result['last_weights']
            print(f"\nFinal Portfolio Weights:")
            print(f"  (Minimum weight constraint: 1% per asset)")
            
            # Count assets at minimum weight
            min_weight_count = sum(1 for w in final_weights.values() if abs(w - 0.01) < 0.001)
            
            for asset, weight in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.01:  # Only show weights > 1%
                    print(f"  {asset}: {weight:.1%}")
            
            if min_weight_count > 0:
                print(f"\n  {min_weight_count} assets at minimum weight (1%)")
    
    # Step 7: Generate visualizations
    print("\n7. Generating visualizations...")
    
    try:
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Black-Litterman Strategy with Upside Views - Performance Analysis', fontsize=16)
        
        # Plot 1: Portfolio value over time
        if 'portfolio_values' in result and len(result['portfolio_values']) > 0:
            portfolio_series = result['portfolio_values']
            axes[0, 0].plot(portfolio_series.index, portfolio_series.values, 
                           linewidth=2, color='blue')
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Asset allocation heatmap
        if 'weight_history' in result:
            weight_df = pd.DataFrame(result['weight_history'])
            # Show only top 10 assets by average weight
            avg_weights = weight_df.mean().sort_values(ascending=False)
            top_assets = avg_weights.head(10).index
            weight_subset = weight_df[top_assets]
            
            sns.heatmap(weight_subset.T, 
                       annot=False,
                       cmap='RdYlBu_r',
                       center=0.5,
                       cbar_kws={'label': 'Weight'},
                       ax=axes[0, 1])
            axes[0, 1].set_title('Asset Allocation Over Time (Top 10 Assets)')
            axes[0, 1].set_xlabel('Rebalancing Date')
            axes[0, 1].set_ylabel('Asset')
        
        # Plot 3: Rolling metrics
        if 'rolling_metrics' in result and not result['rolling_metrics'].empty:
            rolling_metrics = result['rolling_metrics']
            
            if 'rolling_sharpe' in rolling_metrics.columns:
                axes[1, 0].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], 
                               linewidth=2, color='green')
                axes[1, 0].set_title('Rolling Sharpe Ratio')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Sharpe Ratio')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Final weight distribution
        if 'last_weights' in result:
            final_weights = result['last_weights']
            assets = list(final_weights.keys())
            weights = list(final_weights.values())
            
            # Sort by weight
            sorted_data = sorted(zip(assets, weights), key=lambda x: x[1], reverse=True)
            sorted_assets, sorted_weights = zip(*sorted_data)
            
            # Show only top 15 assets
            top_n = min(15, len(sorted_assets))
            axes[1, 1].barh(range(top_n), sorted_weights[:top_n])
            axes[1, 1].set_yticks(range(top_n))
            axes[1, 1].set_yticklabels(sorted_assets[:top_n])
            axes[1, 1].set_title('Final Portfolio Weights (Top 15 Assets)')
            axes[1, 1].set_xlabel('Weight')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('black_litterman_upside_analysis.png', dpi=300, bbox_inches='tight')
        print("   Saved analysis plot to 'black_litterman_upside_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"   Warning: Could not generate visualizations: {e}")
    
    # Step 8: Export results
    print("\n8. Exporting results...")
    
    try:
        # Export detailed results to JSON
        backtester.export_results('black_litterman_upside_results.json', format='json')
        print("   Exported detailed results to 'black_litterman_upside_results.json'")
        
        # Export weight history to CSV
        if 'weight_history' in result:
            weight_df = pd.DataFrame(result['weight_history'])
            weight_df.to_csv('black_litterman_upside_weights.csv')
            print("   Exported weight history to 'black_litterman_upside_weights.csv'")
            
    except Exception as e:
        print(f"   Warning: Could not export results: {e}")
    
    # Step 9: Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"\nStrategy: {strategy.name}")
    print(f"Prior Method: {strategy.prior_method}")
    print(f"View Method: {strategy.view_method}")
    print(f"Minimum Weight: {getattr(strategy, 'min_weight', 0.01):.1%} per asset")
    print(f"Rebalancing: Quarterly (every 3 months)")
    print(f"Performance: {metrics.get('annualized_return', 0):.2%} annualized return")
    print(f"Risk: {metrics.get('volatility', 0):.2%} volatility")
    print(f"Risk-Adjusted: {metrics.get('sharpe_ratio', 0):.3f} Sharpe ratio")
    
    print(f"\nGenerated files:")
    print(f"  - black_litterman_upside_analysis.png")
    print(f"  - black_litterman_upside_results.json")
    print(f"  - black_litterman_upside_weights.csv")
    
    return result


if __name__ == "__main__":
    # Run the example
    result = run_black_litterman_upside_example()
