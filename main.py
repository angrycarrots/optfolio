#!/usr/bin/env python3
"""
Main entry point for the Portfolio Optimization and Backtesting System.

This module provides a simple interface to run portfolio analysis.
"""

import sys
import argparse
from pathlib import Path

from optfolio.data.loader import DataLoader
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory


def main():
    """Main function for the portfolio optimization system."""
    parser = argparse.ArgumentParser(
        description="Portfolio Optimization and Backtesting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample data
  python main.py --sample-data

  # Run with custom data directory
  python main.py --data-dir ./prices --strategies equal_weight mean_variance

  # Run with specific date range
  python main.py --data-dir ./prices --start-date 2021-01-01 --end-date 2023-12-31

  # Run with custom parameters
  python main.py --data-dir ./prices --initial-capital 500000 --rebalance-months 6
        """
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='prices',
        help='Directory containing price data CSV files (default: prices)'
    )
    
    parser.add_argument(
        '--sample-data',
        action='store_true',
        help='Generate and use sample data for demonstration'
    )
    
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['equal_weight', 'mean_variance', 'random_weight', 'black_litterman'],
        choices=['equal_weight', 'mean_variance', 'random_weight', 'black_litterman'],
        help='Strategies to test (default: all strategies)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial portfolio capital (default: 100000)'
    )
    
    parser.add_argument(
        '--rebalance-months',
        type=int,
        default=3,
        help='Rebalancing frequency in months (default: 3)'
    )
    
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.02,
        help='Annual risk-free rate (default: 0.02)'
    )
    
    parser.add_argument(
        '--transaction-costs',
        type=float,
        default=0.001,
        help='Transaction costs as fraction of trade value (default: 0.001)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for results (default: current directory)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    # Handle sample data generation
    if args.sample_data:
        print("Generating sample data...")
        from examples.example import create_sample_data
        data_dir = create_sample_data()
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory '{data_dir}' does not exist")
            print("Use --sample-data to generate sample data for testing")
            sys.exit(1)
    
    try:
        # Initialize data loader
        print(f"Loading data from {data_dir}...")
        data_loader = DataLoader(data_dir)
        
        # Load data
        prices = data_loader.load_prices()
        print(f"Loaded {len(prices.columns)} tickers")
        print(f"Date range: {prices.index.min()} to {prices.index.max()}")
        
        # Create strategies
        print(f"Creating {len(args.strategies)} strategies...")
        strategies = []
        
        strategy_names = {
            'equal_weight': "Equal Weight",
            'mean_variance': "Mean-Variance (Sortino)",
            'random_weight': "Random Weight", 
            'black_litterman': "Black-Litterman"
        }
        
        for strategy_name in args.strategies:
            if strategy_name == 'equal_weight':
                strategy = StrategyFactory.create('equal_weight')
            elif strategy_name == 'mean_variance':
                strategy = StrategyFactory.create('mean_variance', objective="sortino_ratio")
            elif strategy_name == 'random_weight':
                strategy = StrategyFactory.create('random_weight', distribution="dirichlet", seed=42)
            elif strategy_name == 'black_litterman':
                strategy = StrategyFactory.create('black_litterman', 
                                                prior_method="market_cap", 
                                                view_method="momentum")
            
            strategy.name = strategy_names[strategy_name]
            strategies.append(strategy)
            print(f"  Created: {strategy}")
        
        # Initialize backtester
        print("Initializing backtester...")
        backtester = Backtester(
            initial_capital=args.initial_capital,
            risk_free_rate=args.risk_free_rate,
            transaction_costs=args.transaction_costs
        )
        
        backtester.load_data(data_loader)
        
        # Run backtests
        print("Running backtests...")
        rebalance_freq = {"months": args.rebalance_months, "weeks": 1, "days": 1}
        
        results = backtester.run_multiple_backtests(
            strategies=strategies,
            rebalance_freq=rebalance_freq,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        print(f"Completed backtests for {len(results)} strategies")
        
        # Generate comparison
        comparison = backtester.compare_strategies()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Export results
        print("Exporting results...")
        comparison.to_csv(output_dir / 'strategy_comparison.csv', index=False)
        backtester.export_results(output_dir / 'detailed_results.json', format='json')
        
        # Generate plots
        if not args.no_plots:
            try:
                import matplotlib.pyplot as plt
                
                print("Generating plots...")
                
                # Portfolio values plot
                plt.figure(figsize=(12, 8))
                for strategy_name, result in results.items():
                    portfolio_series = result['portfolio_values']
                    plt.plot(portfolio_series.index, portfolio_series.values, 
                            label=strategy_name, linewidth=2)
                
                plt.title('Portfolio Values Over Time')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'portfolio_values.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("  Saved portfolio_values.png")
                
            except ImportError:
                print("  Matplotlib not available, skipping plots")
        
        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print("\nStrategy Performance Comparison:")
        print("-" * 40)
        print(comparison.round(3).to_string(index=False))
        
        # Find best strategy
        if not comparison.empty and 'Sharpe Ratio' in comparison.columns:
            best_strategy = comparison.loc[comparison['Sharpe Ratio'].idxmax(), 'Strategy']
            best_sharpe = comparison['Sharpe Ratio'].max()
            
            print(f"\nBest performing strategy (by Sharpe ratio): {best_strategy}")
            print(f"Sharpe ratio: {best_sharpe:.3f}")
        else:
            print(f"\nNo successful backtests completed. Check the error messages above.")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - strategy_comparison.csv")
        print(f"  - detailed_results.json")
        if not args.no_plots:
            print(f"  - portfolio_values.png")
        
        # Clean up sample data if generated
        if args.sample_data:
            import shutil
            shutil.rmtree(data_dir.parent)
            print("\nSample data cleaned up")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
