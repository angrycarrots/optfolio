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


data_dir = "price"

# Step 2: Load and validate data
print("\n1. Loading and validating data...")
data_loader = DataLoader(data_dir)
sectors = [
        "XLE", "XLF", "XLU", "XLI", "GDX", "XLK", "XLV", "XLY", "XLP", "XLB",
        "XOP", "IYR", "XHB", "ITB", "VNQ", "GDXJ", "IYE", "OIH", "XME", "XRT",
        "SMH", "IBB", "KBE", "KRE", "XTL"
    ]
# Load all available data
prices = data_loader.load_prices(tickers=sectors)
returns = data_loader.get_returns()

print(f"   Loaded {len(prices.columns)} tickers")
print(prices.columns)
print(f"   Date range: {prices.index.min()} to {prices.index.max()}")
print(f"   Total observations: {len(prices)}")

# Validate data
validator = DataValidator()
price_issues = validator.validate_price_data(prices)
returns_issues = validator.validate_returns_data(returns)

print(f"   Data quality score: {validator.get_data_quality_score(prices):.3f}")

if price_issues['warnings']:
    print(f"   Warnings: {len(price_issues['warnings'])}")
if price_issues['errors']:
    print(f"   Errors: {len(price_issues['errors'])}")


# Step 4: Set up backtesting
print("\n3. Setting up backtesting...")

backtester = Backtester(
    initial_capital=100000.0,
    risk_free_rate=0.02,
    transaction_costs=0.001
)

# Load data into backtester
backtester.load_data(data_loader, tickers=sectors)



strategy = StrategyFactory.create('black_litterman', 
                            prior_method="market_cap", view_method="momentum")
rebalance_freq = {"months": 3}                            
result = backtester.run_backtest(strategy=strategy,rebalance_freq=rebalance_freq,
    start_date='2018-01-01',  # Start from 2021 to have enough history
    end_date='2023-12-31')

print(result['rolling_metrics'])

