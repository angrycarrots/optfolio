#!/usr/bin/env python3
"""
Example script demonstrating the portfolio optimization and backtesting system.

This script shows how to:
1. Load data from CSV files in data/price directory
2. Create and configure optimization strategies
3. Run backtests
4. Compare strategy performance
5. Generate reports and visualizations
6. Compare Black-Litterman strategies with different view methods (momentum vs upside)

Data Source: CSV files in data/price directory (not generated sample data)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from pathlib import Path

# Import our modules
from optfolio.data.loader import DataLoader
from optfolio.data.validator import DataValidator
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory


def create_sample_data():
    """Create sample price data for demonstration."""
    print("Creating sample price data...")
    
    # Create temporary directory for sample data
    temp_dir = tempfile.mkdtemp()
    prices_dir = Path(temp_dir) / "prices"
    prices_dir.mkdir()
    
    # Define sector ETFs from the original specification
    sectors = [
        "XLE", "XLF", "XLU", "XLI", "GDX", "XLK", "XLV", "XLY", "XLP", "XLB",
        "XOP", "IYR", "XHB", "ITB", "VNQ", "GDXJ", "IYE", "OIH", "XME", "XRT",
        "SMH", "IBB", "KBE", "KRE", "XTL"
    ]
    
    # Generate realistic price data for each sector
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    for i, ticker in enumerate(sectors):
        # Set different seeds for each ticker to get varied behavior
        np.random.seed(42 + i)
        
        # Generate realistic price series with different characteristics
        base_price = 50 + i * 2  # Different starting prices
        volatility = 0.02 + (i % 5) * 0.005  # Different volatilities
        trend = 0.0001 + (i % 3) * 0.0001  # Different trends
        
        # Generate returns
        returns = np.random.normal(trend, volatility, len(dates))
        
        # Add some market events (crashes, rallies)
        if i % 7 == 0:  # Some sectors have crashes
            crash_indices = np.random.choice(len(dates), size=3, replace=False)
            returns[crash_indices] -= 0.05  # 5% drops
        
        if i % 11 == 0:  # Some sectors have rallies
            rally_indices = np.random.choice(len(dates), size=2, replace=False)
            returns[rally_indices] += 0.03  # 3% gains
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some missing data (simulating real market data)
        missing_indices = np.random.choice(len(dates), size=len(dates)//50, replace=False)
        prices[missing_indices] = np.nan
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        file_path = prices_dir / f"{ticker}.csv"
        df.to_csv(file_path, index=False)
    
    print(f"Created sample data for {len(sectors)} sector ETFs in {prices_dir}")
    return prices_dir


def run_portfolio_analysis():
    """Run the main portfolio analysis."""
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION AND BACKTESTING SYSTEM")
    print("=" * 60)
    
    # Step 1: Use existing CSV data
    # data_dir = create_sample_data()  # Commented out - using existing data
    data_dir = "data/price"  # Path to CSV files
    
    # Step 2: Load and validate data
    print("\n1. Loading and validating data...")
    print(f"   Data directory: {data_dir}")
    
    # Define the specific symbols to analyze
    target_symbols = [
        'WSM', 'PAYX', 'BMI', 'BK', 'NDAQ', 'MSI', 'WMT', 'TJX', 'AIG', 'RJF', 
        'V', 'CTAS', 'TT', 'TRGP', 'JPM', 'GE', 'MCK', 'PH', 'LLY', 'COST', 
        'AVGO', 'NEE', 'AMAT', 'ADI', 'SHW', 'INTU', 'KLAC'
    ]
    
    print(f"   🎯 Target symbols: {len(target_symbols)} tickers")
    print(f"   📋 Symbol list: {', '.join(target_symbols)}")
    
    try:
        data_loader = DataLoader(data_dir)
        
        # Load data for specific symbols only
        prices = data_loader.load_prices(tickers=target_symbols)
        returns = data_loader.get_returns()
        
        # Filter returns to match the loaded prices
        returns = returns[prices.columns]
        
        print(f"   ✅ Successfully loaded {len(prices.columns)} target tickers")
        print(f"   📅 Date range: {prices.index.min()} to {prices.index.max()}")
        print(f"   📊 Total observations: {len(prices)}")
        print(f"   📈 Loaded tickers: {', '.join(prices.columns)}")
        
        # Check if any target symbols were not found
        missing_symbols = set(target_symbols) - set(prices.columns)
        if missing_symbols:
            print(f"   ⚠️  Warning: {len(missing_symbols)} symbols not found: {', '.join(missing_symbols)}")
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        print(f"   💡 Make sure the data directory '{data_dir}' contains CSV files for the target symbols")
        raise
    
    # Validate data
    validator = DataValidator()
    price_issues = validator.validate_price_data(prices)
    returns_issues = validator.validate_returns_data(returns)
    
    print(f"   Data quality score: {validator.get_data_quality_score(prices):.3f}")
    
    if price_issues['warnings']:
        print(f"   Warnings: {len(price_issues['warnings'])}")
    if price_issues['errors']:
        print(f"   Errors: {len(price_issues['errors'])}")
    
    # Step 3: Create optimization strategies
    print("\n2. Creating optimization strategies...")
    
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
    
    strategy_names = ["Equal Weight", "Mean-Variance (Sortino)", "Mean-Variance (Sharpe)", "Random Weight", "Black-Litterman (Momentum)", "Black-Litterman (Upside)"]
    for i, strategy in enumerate(strategies):
        strategy.name = strategy_names[i]
        
        # Set minimum weight constraint for Black-Litterman strategies
        if 'black_litterman' in strategy.name.lower():
            strategy.min_weight = 0.01  # 1% minimum allocation
            print(f"   Created: {strategy} (min weight: 1%)")
        else:
            print(f"   Created: {strategy}")
    
    # Step 4: Set up backtesting
    print("\n3. Setting up backtesting...")
    
    backtester = Backtester(
        initial_capital=100000.0,
        risk_free_rate=0.02,
        transaction_costs=0.001
    )
    
    # Load data into backtester (using the same filtered symbols)
    backtester.load_data(data_loader, tickers=prices.columns.tolist())
    
    # Step 5: Run backtests
    print("\n4. Running backtests...")
    
    # Define rebalancing schedule (every 3 months, 1st week, 1st day)
    rebalance_freq = {"months": 3, "weeks": 1, "days": 1}
    
    # Run backtests for all strategies
    results = backtester.run_multiple_backtests(
        strategies=strategies,
        rebalance_freq=rebalance_freq,
        start_date='2021-01-01',  # Start from 2021 to have enough history
        end_date='2023-12-31'
    )
    
    print(f"   Completed backtests for {len(results)} strategies")
    
    # Step 6: Compare strategies
    print("\n5. Comparing strategy performance...")
    
    comparison = backtester.compare_strategies()
    print("\nStrategy Performance Comparison:")
    print("-" * 80)
    print(comparison.round(3).to_string(index=False))
    
    # Step 7: Generate detailed analysis
    print("\n6. Detailed strategy analysis...")
    
    # Track Black-Litterman strategies for comparison
    bl_strategies = {}
    
    for strategy_name, result in results.items():
        metrics = result['performance_metrics']
        summary = result['summary']
        
        print(f"\n{strategy_name}:")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Number of Transactions: {summary.get('num_transactions', 0)}")
        print(f"  Total Transaction Costs: ${summary.get('total_transaction_costs', 0):.2f}")
        
        # Store Black-Litterman results for comparison
        if 'Black-Litterman' in strategy_name:
            bl_strategies[strategy_name] = result
    
    # Compare Black-Litterman strategies if we have both
    if len(bl_strategies) >= 2:
        print(f"\n" + "="*60)
        print("BLACK-LITTERMAN STRATEGY COMPARISON")
        print("="*60)
        
        for strategy_name, result in bl_strategies.items():
            metrics = result['performance_metrics']
            summary = result['summary']
            
            print(f"\n{strategy_name}:")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Transaction Costs: ${summary.get('total_transaction_costs', 0):.2f}")
            
            # Show asset allocation if available
            if 'last_weights' in result:
                final_weights = result['last_weights']
                top_assets = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top 5 Assets:")
                for asset, weight in top_assets:
                    print(f"    {asset}: {weight:.1%}")
    
    # Step 8: Export results
    print("\n7. Exporting results...")
    
    # Export comparison to CSV
    comparison.to_csv('strategy_comparison.csv', index=False)
    print("   Exported strategy comparison to 'strategy_comparison.csv'")
    
    # Export detailed results to JSON
    backtester.export_results('detailed_results.json', format='json')
    print("   Exported detailed results to 'detailed_results.json'")
    
    # Step 9: Generate visualizations
    print("\n8. Generating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Plot portfolio values
        plt.figure(figsize=(12, 8))
        for strategy_name, result in results.items():
            if 'portfolio_values' in result and len(result['portfolio_values']) > 0:
                portfolio_series = result['portfolio_values']
                plt.plot(portfolio_series.index, portfolio_series.values, 
                        label=strategy_name, linewidth=2)
        
        plt.title('Portfolio Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('portfolio_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   Saved portfolio values plot to 'portfolio_values.png'")
        
        # Plot rolling metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (strategy_name, result) in enumerate(results.items()):
            rolling_metrics = result['rolling_metrics']
            
            if not rolling_metrics.empty:
                # Rolling Sharpe ratio
                axes[0, 0].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], 
                               label=strategy_name, alpha=0.7)
                
                # Rolling volatility
                axes[0, 1].plot(rolling_metrics.index, rolling_metrics['rolling_volatility'], 
                               label=strategy_name, alpha=0.7)
                
                # Rolling return
                axes[1, 0].plot(rolling_metrics.index, rolling_metrics['rolling_return'], 
                               label=strategy_name, alpha=0.7)
                
                # Rolling drawdown
                axes[1, 1].plot(rolling_metrics.index, rolling_metrics['rolling_drawdown'], 
                               label=strategy_name, alpha=0.7)
        
        axes[0, 0].set_title('Rolling Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Rolling Volatility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Rolling Return')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Rolling Drawdown')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rolling_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   Saved rolling metrics plot to 'rolling_metrics.png'")
        
        # Additional plot: Compare Black-Litterman strategies
        if len(bl_strategies) >= 2:
            plt.figure(figsize=(15, 10))
            
            # Create subplots for different comparisons
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Black-Litterman Strategy Comparison', fontsize=16)
            
            # Plot 1: Portfolio values comparison
            for strategy_name, result in bl_strategies.items():
                if 'portfolio_values' in result and len(result['portfolio_values']) > 0:
                    portfolio_series = result['portfolio_values']
                    axes[0, 0].plot(portfolio_series.index, portfolio_series.values, 
                                   label=strategy_name, linewidth=2)
            
            axes[0, 0].set_title('Portfolio Values Comparison')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Rolling Sharpe ratio comparison
            for strategy_name, result in bl_strategies.items():
                if 'rolling_metrics' in result and not result['rolling_metrics'].empty:
                    rolling_metrics = result['rolling_metrics']
                    if 'rolling_sharpe' in rolling_metrics.columns:
                        axes[0, 1].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], 
                                       label=strategy_name, linewidth=2)
            
            axes[0, 1].set_title('Rolling Sharpe Ratio Comparison')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Weight distribution comparison
            for i, (strategy_name, result) in enumerate(bl_strategies.items()):
                if 'last_weights' in result:
                    final_weights = result['last_weights']
                    assets = list(final_weights.keys())
                    weights = list(final_weights.values())
                    
                    # Sort by weight and show top 10
                    sorted_data = sorted(zip(assets, weights), key=lambda x: x[1], reverse=True)
                    sorted_assets, sorted_weights = zip(*sorted_data)
                    
                    top_n = min(10, len(sorted_assets))
                    x_pos = np.arange(top_n) + i * 0.4
                    axes[1, 0].bar(x_pos, sorted_weights[:top_n], width=0.4, 
                                   label=strategy_name, alpha=0.7)
            
            axes[1, 0].set_title('Top 10 Asset Weights Comparison')
            axes[1, 0].set_xlabel('Asset Rank')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_xticks(np.arange(10) + 0.2)
            axes[1, 0].set_xticklabels([f'#{i+1}' for i in range(10)])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Performance metrics comparison
            metrics_names = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
            x_pos = np.arange(len(metrics_names))
            width = 0.35
            
            for i, (strategy_name, result) in enumerate(bl_strategies.items()):
                metrics = result['performance_metrics']
                values = [
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('sortino_ratio', 0),
                    abs(metrics.get('max_drawdown', 0))  # Use absolute value for drawdown
                ]
                
                x_offset = i * width
                axes[1, 1].bar(x_pos + x_offset, values, width, label=strategy_name, alpha=0.7)
            
            axes[1, 1].set_title('Performance Metrics Comparison')
            axes[1, 1].set_xlabel('Metric')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_xticks(x_pos + width/2)
            axes[1, 1].set_xticklabels(metrics_names)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('black_litterman_comparison.png', dpi=300, bbox_inches='tight')
            print("   Saved Black-Litterman comparison plot to 'black_litterman_comparison.png'")
            plt.show()
            plt.close()
        
    except ImportError:
        print("   Matplotlib not available, skipping visualizations")
    
    # Step 10: Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Find best performing strategy
    if not comparison.empty and 'Sharpe Ratio' in comparison.columns:
        best_strategy = comparison.loc[comparison['Sharpe Ratio'].idxmax(), 'Strategy']
        best_sharpe = comparison['Sharpe Ratio'].max()
        
        print(f"\nBest performing strategy (by Sharpe ratio): {best_strategy}")
        print(f"Sharpe ratio: {best_sharpe:.3f}")
    else:
        print(f"\nNo successful backtests completed. Check the error messages above.")
    
    # Clean up temporary data (only if it's a temporary directory)
    import shutil
    if isinstance(data_dir, Path) and data_dir.parent.exists():
        try:
            shutil.rmtree(data_dir.parent)
            print(f"\nTemporary data cleaned up")
        except Exception as e:
            print(f"\nWarning: Could not clean up temporary data: {e}")
    else:
        print(f"\nUsing existing CSV data from: {data_dir}")
        print(f"   Analyzed {len(prices.columns)} target tickers from {len(target_symbols)} requested")
        print(f"   Target symbols: {', '.join(target_symbols)}")
    print(f"\nGenerated files:")
    print(f"  - strategy_comparison.csv")
    print(f"  - detailed_results.json")
    print(f"  - portfolio_values.png")
    print(f"  - rolling_metrics.png")
    if len(bl_strategies) >= 2:
        print(f"  - black_litterman_comparison.png")


if __name__ == "__main__":
    run_portfolio_analysis()
