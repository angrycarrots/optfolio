#!/usr/bin/env python3
"""
Test script for upside views in Black-Litterman strategy.
"""

import sys
import os

# Add the optfolio directory to the path
sys.path.append('optfolio')

try:
    from strategies.upside import UpsideCalculator
    from strategies.black_litterman import BlackLittermanStrategy
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_upside_calculator():
    """Test the UpsideCalculator class."""
    print("\n" + "="*50)
    print("Testing UpsideCalculator")
    print("="*50)
    
    try:
        # Initialize calculator
        uc = UpsideCalculator()
        print(f"✅ UpsideCalculator initialized with data directory: {uc.datadir}")
        
        # Check if data directory exists
        if uc.datadir.exists():
            print(f"✅ Data directory exists: {uc.datadir}")
            
            # Get available symbols
            symbols = uc.get_available_symbols()
            print(f"✅ Found {len(symbols)} symbols: {symbols[:5]}...")
            
            # Test with first available symbol
            if symbols:
                test_symbol = symbols[0]
                print(f"\nTesting with symbol: {test_symbol}")
                
                try:
                    upside_data = uc.upside(test_symbol)
                    print(f"✅ Successfully calculated upside for {test_symbol}")
                    print(f"   Data shape: {upside_data.shape}")
                    print(f"   Columns: {list(upside_data.columns)}")
                    
                    if not upside_data.empty:
                        latest = upside_data.iloc[-1]
                        print(f"   Latest data:")
                        print(f"     Close: {latest['close']}")
                        print(f"     Price Target: {latest['price_target']}")
                        print(f"     Upside: {latest['upside']:.4f}")
                    else:
                        print("   ⚠️ No upside data available")
                        
                except Exception as e:
                    print(f"❌ Error calculating upside for {test_symbol}: {e}")
            else:
                print("⚠️ No symbols found in data directory")
        else:
            print(f"❌ Data directory does not exist: {uc.datadir}")
            
    except Exception as e:
        print(f"❌ Error testing UpsideCalculator: {e}")

def test_black_litterman_upside():
    """Test the Black-Litterman strategy with upside views."""
    print("\n" + "="*50)
    print("Testing Black-Litterman with Upside Views")
    print("="*50)
    
    try:
        # Create strategy
        strategy = BlackLittermanStrategy(
            name="Test BL Upside",
            prior_method="market_cap",
            view_method="upside"
        )
        print(f"✅ Strategy created: {strategy}")
        print(f"   Prior method: {strategy.prior_method}")
        print(f"   View method: {strategy.view_method}")
        
        # Test view generation (we'll need some sample data)
        print("\nTesting view generation...")
        
        # Create sample returns data
        import pandas as pd
        import numpy as np
        
        # Sample data with some of the symbols from your example
        sample_symbols = ['WSM', 'PAYX', 'BMI', 'BK', 'NDAQ']
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Create sample returns (random walk)
        np.random.seed(42)
        returns_data = {}
        for symbol in sample_symbols:
            returns_data[symbol] = np.random.normal(0.0001, 0.02, len(dates))
        
        returns_df = pd.DataFrame(returns_data, index=dates)
        print(f"✅ Created sample returns data: {returns_df.shape}")
        
        # Test view generation
        try:
            views, confidence = strategy._generate_upside_views(returns_df)
            print(f"✅ Successfully generated views")
            print(f"   Views matrix shape: {views.shape}")
            print(f"   Confidence levels: {confidence}")
            
            # Show some view details
            for i, symbol in enumerate(sample_symbols):
                if i < len(confidence):
                    print(f"   {symbol}: confidence = {confidence[i]:.3f}")
                    
        except Exception as e:
            print(f"❌ Error generating views: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error testing Black-Litterman strategy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Upside Views Test Suite")
    
    # Test 1: UpsideCalculator
    test_upside_calculator()
    
    # Test 2: Black-Litterman with upside views
    test_black_litterman_upside()
    
    print("\n" + "="*50)
    print("Test Suite Complete")
    print("="*50)
