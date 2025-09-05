#!/usr/bin/env python3
"""
Test script to verify Black-Litterman strategy fixes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from optfolio.data.loader import DataLoader
from optfolio.strategies.base import StrategyFactory

def test_black_litterman():
    """Test Black-Litterman strategy with numerical stability fixes."""
    print("Testing Black-Litterman strategy...")
    
    # Load data
    data_dir = "price"
    data_loader = DataLoader(data_dir)
    prices = data_loader.load_prices()
    returns = data_loader.get_returns()
    
    print(f"Loaded {len(prices.columns)} assets")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    # Create Black-Litterman strategy
    strategy = StrategyFactory.create('black_litterman', 
                                    prior_method="market_cap", 
                                    view_method="momentum")
    strategy.name = "Black-Litterman Test"
    
    print(f"Created strategy: {strategy}")
    
    # Test optimization on a subset of data
    test_returns = returns.tail(252)  # Last year of data
    print(f"Testing with {len(test_returns)} days of data")
    
    try:
        # Optimize weights
        weights = strategy.optimize(test_returns)
        
        print("Optimization successful!")
        print(f"Number of assets: {len(weights)}")
        print(f"Weight sum: {sum(weights.values()):.6f}")
        print(f"Min weight: {min(weights.values()):.6f}")
        print(f"Max weight: {max(weights.values()):.6f}")
        
        # Check for any invalid values
        invalid_weights = [w for w in weights.values() if not np.isfinite(w)]
        if invalid_weights:
            print(f"Warning: Found {len(invalid_weights)} invalid weights")
        else:
            print("All weights are valid!")
            
        return True
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        return False

def test_multiple_black_litterman_configurations():
    """Test different Black-Litterman configurations."""
    print("\n" + "="*60)
    print("Testing Multiple Black-Litterman Configurations")
    print("="*60)
    
    # Load data
    data_dir = "price"
    data_loader = DataLoader(data_dir)
    data_loader.load_prices()  # Load prices first
    returns = data_loader.get_returns().tail(252)  # Last year of data
    
    configurations = [
        {"prior_method": "market_cap", "view_method": "momentum", "name": "Market Cap + Momentum"},
        {"prior_method": "market_cap", "view_method": "mean_reversion", "name": "Market Cap + Mean Reversion"},
        {"prior_method": "market_cap", "view_method": "random", "name": "Market Cap + Random Views"},
        {"prior_method": "equal", "view_method": "momentum", "name": "Equal + Momentum"},
        {"prior_method": "random", "view_method": "momentum", "name": "Random + Momentum"},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        try:
            strategy = StrategyFactory.create('black_litterman', 
                                            prior_method=config["prior_method"], 
                                            view_method=config["view_method"])
            strategy.name = config["name"]
            
            weights = strategy.optimize(returns)
            
            # Calculate basic portfolio metrics
            portfolio_return = sum(weights[ticker] * returns[ticker].mean() for ticker in weights)
            portfolio_vol = np.sqrt(sum(weights[ticker] * weights[ticker2] * returns[ticker].cov(returns[ticker2]) 
                                       for ticker in weights for ticker2 in weights))
            
            results[config['name']] = {
                'weights': weights,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            }
            
            print(f"  ✅ Success - Return: {portfolio_return:.4f}, Vol: {portfolio_vol:.4f}, Sharpe: {results[config['name']]['sharpe']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[config['name']] = None
    
    return results

def test_black_litterman_parameters():
    """Test Black-Litterman with different parameter settings."""
    print("\n" + "="*60)
    print("Testing Black-Litterman Parameters")
    print("="*60)
    
    # Load data
    data_dir = "price"
    data_loader = DataLoader(data_dir)
    data_loader.load_prices()  # Load prices first
    returns = data_loader.get_returns().tail(252)  # Last year of data
    
    # Test different tau values
    tau_values = [0.01, 0.05, 0.1, 0.2]
    risk_aversion_values = [1.0, 2.0, 3.0, 5.0]
    
    print("\nTesting different tau values:")
    for tau in tau_values:
        try:
            strategy = StrategyFactory.create('black_litterman', 
                                            prior_method="market_cap", 
                                            view_method="momentum",
                                            tau=tau)
            weights = strategy.optimize(returns)
            print(f"  Tau={tau}: ✅ Success")
        except Exception as e:
            print(f"  Tau={tau}: ❌ Failed - {e}")
    
    print("\nTesting different risk aversion values:")
    for risk_aversion in risk_aversion_values:
        try:
            strategy = StrategyFactory.create('black_litterman', 
                                            prior_method="market_cap", 
                                            view_method="momentum",
                                            risk_aversion=risk_aversion)
            weights = strategy.optimize(returns)
            print(f"  Risk Aversion={risk_aversion}: ✅ Success")
        except Exception as e:
            print(f"  Risk Aversion={risk_aversion}: ❌ Failed - {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("BLACK-LITTERMAN STRATEGY TESTING")
    print("=" * 60)
    
    # Basic functionality test
    success = test_black_litterman()
    if success:
        print("\n✅ Basic Black-Litterman test passed!")
    else:
        print("\n❌ Basic Black-Litterman test failed!")
    
    # Test multiple configurations
    config_results = test_multiple_black_litterman_configurations()
    
    # Test different parameters
    test_black_litterman_parameters()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    
    # Summary
    successful_configs = sum(1 for result in config_results.values() if result is not None)
    total_configs = len(config_results)
    
    print(f"\nConfiguration Test Results:")
    print(f"  Successful: {successful_configs}/{total_configs}")
    print(f"  Success Rate: {successful_configs/total_configs*100:.1f}%")
    
    if successful_configs == total_configs:
        print("\n🎉 All Black-Litterman configurations working correctly!")
    else:
        print(f"\n⚠️  {total_configs - successful_configs} configurations failed")
