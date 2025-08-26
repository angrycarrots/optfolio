from optfolio.data.loader import DataLoader
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory

# Load data
loader = DataLoader("price")
prices = loader.load_prices()

# Create strategies
equal_weight = StrategyFactory.create('equal_weight')
mean_variance = StrategyFactory.create('mean_variance', objective='sortino_ratio')

# Run backtest
backtester = Backtester(initial_capital=100000)
backtester.load_data(loader)
results = backtester.run_multiple_backtests([equal_weight, mean_variance])