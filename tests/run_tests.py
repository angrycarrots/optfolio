#!/usr/bin/env python3
"""
Test runner for all buy-and-hold fix tests.
"""

import sys
import os
import unittest
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_test_suite():
    """Run all test suites."""
    print("🧪 Running Buy-and-Hold Fix Test Suite")
    print("=" * 50)
    
    # Test suites to run
    test_suites = [
        ('Portfolio Metrics Tests', 'test_portfolio_metrics'),
        ('Backtesting Buy-and-Hold Tests', 'test_backtesting_buy_and_hold'),
        ('Streamlit Fix Tests', 'test_streamlit_fix'),
        ('Integration Tests', 'test_integration'),
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for suite_name, module_name in test_suites:
        print(f"\n📋 Running {suite_name}...")
        print("-" * 30)
        
        try:
            # Import and run the test module
            module = __import__(module_name)
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            result = runner.run(suite)
            
            # Accumulate results
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            
            # Print summary for this suite
            if result.wasSuccessful():
                print(f"✅ {suite_name}: PASSED ({result.testsRun} tests)")
            else:
                print(f"❌ {suite_name}: FAILED ({result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors)")
                
        except ImportError as e:
            print(f"❌ {suite_name}: Could not import module - {e}")
            total_errors += 1
        except Exception as e:
            print(f"❌ {suite_name}: Unexpected error - {e}")
            total_errors += 1
    
    # Print overall summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    
    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("💥 SOME TESTS FAILED!")
        return False

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("\n🚀 Running Quick Smoke Test...")
    print("-" * 30)
    
    try:
        from optfolio.backtesting.engine import Backtester
        from optfolio.strategies.base import StrategyFactory
        from optfolio.data.loader import DataLoader
        
        # Load minimal data
        data_loader = DataLoader("data/price")
        test_symbols = ['WSM', 'PAYX']  # Just 2 symbols for speed
        prices = data_loader.load_prices(tickers=test_symbols)
        returns = data_loader.get_returns()
        returns = returns[prices.columns]
        
        # Create backtester
        backtester = Backtester(
            initial_capital=100000.0,
            risk_free_rate=0.02,
            transaction_costs=0.001
        )
        backtester.load_data(data_loader, tickers=prices.columns.tolist())
        
        # Test buy-and-hold strategy
        strategy = StrategyFactory.create('buy_and_hold', allocation_method="equal_weight")
        strategy.name = "Buy and Hold"
        
        results = backtester.run_backtest(
            strategy=strategy,
            rebalance_freq=None,
            start_date='2023-01-01',
            end_date='2023-03-31'  # Short period for speed
        )
        
        # Verify basic results
        summary = results['summary']
        assert summary['num_rebalances'] == 1, f"Expected 1 rebalance, got {summary['num_rebalances']}"
        assert summary['num_transactions'] == 2, f"Expected 2 transactions, got {summary['num_transactions']}"
        
        print("✅ Smoke test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    start_time = time.time()
    
    # Run smoke test first
    smoke_success = run_quick_smoke_test()
    
    if smoke_success:
        # Run full test suite
        success = run_test_suite()
    else:
        print("❌ Smoke test failed, skipping full test suite")
        success = False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
