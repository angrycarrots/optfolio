#!/usr/bin/env python3
"""
Test script for minimum weight constraints in Black-Litterman strategy.
"""

import sys
import numpy as np
import pandas as pd

# Add the optfolio directory to the path
sys.path.append('optfolio')

try:
    from strategies.black_litterman import BlackLittermanStrategy
    print("✅ Successfully imported BlackLittermanStrategy")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_minimum_weights():
    """Test the minimum weight constraint functionality."""
    print("\n" + "="*60)
    print("Testing Minimum Weight Constraints")
    print("="*60)
    
    # Create sample returns data
    np.random.seed(42)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generate sample returns
    returns_data = {}
    for symbol in symbols:
        returns_data[symbol] = np.random.normal(0.0001, 0.02, len(dates))
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    print(f"✅ Created sample returns data: {returns_df.shape}")
    
    # Test different minimum weight constraints
    min_weights = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
    
    for min_weight in min_weights:
        print(f"\n--- Testing with minimum weight: {min_weight:.1%} ---")
        
        try:
            # Create strategy
            strategy = BlackLittermanStrategy(
                name=f"Test BL (min={min_weight:.1%})",
                prior_method="equal",
                view_method="random"
            )
            
            # Optimize with minimum weight constraint
            weights = strategy.optimize(returns_df, min_weight=min_weight)
            
            print(f"✅ Optimization successful")
            print(f"   Number of assets: {len(weights)}")
            print(f"   Total weight: {sum(weights.values()):.6f}")
            
            # Check minimum weight constraint
            min_actual = min(weights.values())
            max_actual = max(weights.values())
            
            print(f"   Weight range: {min_actual:.1%} to {max_actual:.1%}")
            
            if min_actual >= min_weight:
                print(f"   ✅ Minimum weight constraint satisfied: {min_actual:.1%} >= {min_weight:.1%}")
            else:
                print(f"   ❌ Minimum weight constraint violated: {min_actual:.1%} < {min_weight:.1%}")
            
            # Show weight distribution
            print(f"   Weight distribution:")
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for asset, weight in sorted_weights:
                print(f"     {asset}: {weight:.1%}")
            
        except Exception as e:
            print(f"❌ Error with min_weight={min_weight}: {e}")
            import traceback
            traceback.print_exc()

def test_weight_enforcement():
    """Test the weight enforcement method directly."""
    print("\n" + "="*60)
    print("Testing Weight Enforcement Method")
    print("="*60)
    
    try:
        strategy = BlackLittermanStrategy()
        
        # Test case 1: Some weights below minimum
        test_weights = {
            'AAPL': 0.005,   # 0.5% - below minimum
            'MSFT': 0.015,    # 1.5% - above minimum
            'GOOGL': 0.008,   # 0.8% - below minimum
            'AMZN': 0.972     # 97.2% - above minimum
        }
        
        print(f"Original weights: {test_weights}")
        print(f"Total: {sum(test_weights.values()):.3f}")
        
        # Enforce minimum weight
        enforced_weights = strategy._enforce_minimum_weights(test_weights, min_weight=0.01)
        
        print(f"\nEnforced weights: {enforced_weights}")
        print(f"Total: {sum(enforced_weights.values()):.3f}")
        print(f"Min weight: {min(enforced_weights.values()):.1%}")
        print(f"Max weight: {max(enforced_weights.values()):.1%}")
        
        # Verify constraints
        min_satisfied = min(enforced_weights.values()) >= 0.01
        sum_satisfied = abs(sum(enforced_weights.values()) - 1.0) < 1e-6
        
        print(f"\nConstraints satisfied:")
        print(f"  Minimum weight: {'✅' if min_satisfied else '❌'}")
        print(f"  Sum to 1: {'✅' if sum_satisfied else '❌'}")
        
    except Exception as e:
        print(f"❌ Error testing weight enforcement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Minimum Weight Test Suite")
    
    # Test 1: Minimum weight constraints in optimization
    test_minimum_weights()
    
    # Test 2: Weight enforcement method
    test_weight_enforcement()
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)
