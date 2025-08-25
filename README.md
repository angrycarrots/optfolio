# Portfolio Optimization and Backtesting System

A comprehensive Python system for portfolio optimization, backtesting, and performance analysis using multiple optimization strategies.

## Features

- **Multiple Optimization Strategies**: Equal weight, Mean-variance (Sortino/Sharpe), Random weight, and Black-Litterman
- **Flexible Rebalancing**: Configurable rebalancing schedules (N_months, N_weeks, N_days)
- **Comprehensive Metrics**: Returns, drawdown, Sharpe ratio, Sortino ratio, Beta, VaR, CVaR, and more
- **Data Validation**: Robust data loading and validation with quality scoring
- **Transaction Costs**: Realistic transaction cost modeling
- **Visualization**: Portfolio performance charts and rolling metrics
- **Export Capabilities**: Results export to CSV, Excel, and JSON formats

## Installation

### Prerequisites

- Python 3.12 or higher
- uv package manager (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd optfolio
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

## Quick Start

### Run with Sample Data

The easiest way to get started is to run the system with sample data:

```bash
python main.py --sample-data
```

This will:
- Generate realistic sample data for 25 sector ETFs
- Run all four optimization strategies
- Generate performance comparisons and visualizations
- Export results to CSV and JSON files

### Run with Your Own Data

1. Prepare your data in CSV format:
   - Create a `prices` directory
   - Add CSV files named `{TICKER}.csv` (e.g., `AAPL.csv`, `GOOGL.csv`)
   - Each CSV should have columns: `date`, `close`

2. Run the analysis:
```bash
python main.py --data-dir ./prices --strategies equal_weight mean_variance
```

## Data Format

### Price Data Files

Each ticker should have a CSV file with the following structure:

```csv
date,close
2020-01-01,100.50
2020-01-02,101.20
2020-01-03,99.80
...
```

### Supported Sector ETFs

The system is designed to work with the following sector ETFs:

- **Energy**: XLE, XOP, IYE, OIH
- **Financials**: XLF, KBE, KRE
- **Utilities**: XLU
- **Industrials**: XLI, XME
- **Technology**: XLK, SMH, XTL
- **Healthcare**: XLV, IBB
- **Consumer Discretionary**: XLY, XRT
- **Consumer Staples**: XLP
- **Materials**: XLB
- **Real Estate**: IYR, XHB, ITB, VNQ
- **Mining**: GDX, GDXJ

## Optimization Strategies

### 1. Equal Weight Strategy
- Allocates equal weights to all assets
- Simple and effective baseline strategy
- No optimization required

### 2. Mean-Variance Strategy
- Maximizes risk-adjusted returns
- Objectives: Sortino ratio, Sharpe ratio, minimum variance, maximum return
- Uses scipy optimization with fallback to skfolio

### 3. Random Weight Strategy
- Generates random portfolio weights
- Distributions: Dirichlet, Uniform, Normal
- Useful for Monte Carlo analysis

### 4. Black-Litterman Strategy
- Combines market equilibrium with investor views
- Prior methods: Market cap, Equal, Random
- View methods: Random, Momentum, Mean reversion

## Command Line Interface

### Basic Usage

```bash
python main.py [OPTIONS]
```

### Options

- `--data-dir DIR`: Directory containing price data (default: prices)
- `--sample-data`: Generate and use sample data
- `--strategies LIST`: Strategies to test (choices: equal_weight, mean_variance, random_weight, black_litterman)
- `--start-date DATE`: Start date for backtest (YYYY-MM-DD)
- `--end-date DATE`: End date for backtest (YYYY-MM-DD)
- `--initial-capital AMOUNT`: Initial portfolio capital (default: 100000)
- `--rebalance-months MONTHS`: Rebalancing frequency in months (default: 3)
- `--risk-free-rate RATE`: Annual risk-free rate (default: 0.02)
- `--transaction-costs COSTS`: Transaction costs as fraction (default: 0.001)
- `--output-dir DIR`: Output directory for results (default: current directory)
- `--no-plots`: Skip generating plots

### Examples

```bash
# Run with sample data
python main.py --sample-data

# Test specific strategies
python main.py --data-dir ./prices --strategies equal_weight mean_variance

# Custom parameters
python main.py --data-dir ./prices --initial-capital 500000 --rebalance-months 6

# Specific date range
python main.py --data-dir ./prices --start-date 2021-01-01 --end-date 2023-12-31

# No visualizations
python main.py --sample-data --no-plots
```

## Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- Total Return
- Annualized Return
- Rolling Returns

### Risk Metrics
- Volatility (Annualized)
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR)

### Risk-Adjusted Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Treynor Ratio
- Information Ratio

### Other Metrics
- Beta (vs market)
- Rolling Beta
- Transaction Costs
- Number of Trades

## Output Files

The system generates several output files:

- `strategy_comparison.csv`: Performance comparison table
- `detailed_results.json`: Detailed backtest results
- `portfolio_values.png`: Portfolio value over time chart
- `rolling_metrics.png`: Rolling performance metrics

## API Usage

### Basic Example

```python
from optfolio.data.loader import DataLoader
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory

# Load data
data_loader = DataLoader("prices")
prices = data_loader.load_prices()

# Create strategies
equal_weight = StrategyFactory.create('equal_weight')
mean_variance = StrategyFactory.create('mean_variance', objective='sortino_ratio')

# Run backtest
backtester = Backtester(initial_capital=100000)
backtester.load_data(data_loader)

results = backtester.run_multiple_backtests(
    strategies=[equal_weight, mean_variance],
    rebalance_freq={"months": 3, "weeks": 1, "days": 1}
)

# Compare results
comparison = backtester.compare_strategies()
print(comparison)
```

### Advanced Example

```python
from optfolio.strategies.black_litterman import BlackLittermanStrategy
from optfolio.portfolio.metrics import PortfolioMetrics

# Create custom Black-Litterman strategy
bl_strategy = BlackLittermanStrategy(
    tau=0.05,
    risk_aversion=2.5,
    prior_method="market_cap",
    view_method="momentum"
)

# Run backtest with custom parameters
results = backtester.run_backtest(
    strategy=bl_strategy,
    rebalance_freq={"months": 6},
    start_date="2021-01-01",
    end_date="2023-12-31"
)

# Calculate additional metrics
metrics = PortfolioMetrics(risk_free_rate=0.02)
performance = metrics.calculate_all_metrics(results['portfolio_values'])
print(performance)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=optfolio

# Run specific test modules
pytest optfolio/tests/test_data.py
pytest optfolio/tests/test_strategies.py
```

## Project Structure

```
optfolio/
├── data/                   # Data loading and validation
│   ├── loader.py          # CSV data loader
│   └── validator.py       # Data validation
├── portfolio/             # Portfolio management
│   ├── base.py           # Portfolio class
│   └── metrics.py        # Performance metrics
├── strategies/           # Optimization strategies
│   ├── base.py          # Strategy base class
│   ├── equal_weight.py  # Equal weight strategy
│   ├── mean_variance.py # Mean-variance strategy
│   ├── random_weight.py # Random weight strategy
│   └── black_litterman.py # Black-Litterman strategy
├── backtesting/         # Backtesting engine
│   └── engine.py       # Main backtester
├── analysis/           # Analysis and visualization
├── tests/             # Test suite
├── cli/               # Command line interface
└── config/            # Configuration management
```

## Configuration

### Rebalancing Schedule

The system supports flexible rebalancing schedules:

```python
# Every 3 months, 1st week, 1st day
rebalance_freq = {"months": 3, "weeks": 1, "days": 1}

# Every 6 months
rebalance_freq = {"months": 6}

# Every week
rebalance_freq = {"weeks": 1}

# Every day
rebalance_freq = {"days": 1}
```

### Strategy Parameters

Each strategy can be customized:

```python
# Mean-variance with different objectives
mv_sortino = StrategyFactory.create('mean_variance', objective='sortino_ratio')
mv_sharpe = StrategyFactory.create('mean_variance', objective='sharpe_ratio')
mv_minvar = StrategyFactory.create('mean_variance', objective='min_variance')

# Random weight with different distributions
rw_dirichlet = StrategyFactory.create('random_weight', distribution='dirichlet')
rw_uniform = StrategyFactory.create('random_weight', distribution='uniform')
rw_normal = StrategyFactory.create('random_weight', distribution='normal')

# Black-Litterman with different priors and views
bl_momentum = StrategyFactory.create('black_litterman', 
                                   prior_method='market_cap', 
                                   view_method='momentum')
bl_meanrev = StrategyFactory.create('black_litterman',
                                  prior_method='equal',
                                  view_method='mean_reversion')
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VectorBT for backtesting infrastructure
- Skfolio for portfolio optimization
- Pandas and NumPy for data manipulation
- Matplotlib for visualization

## Support

For questions and support:
- Check the documentation
- Run the example scripts
- Review the test cases
- Open an issue on GitHub
