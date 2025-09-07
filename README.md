# Portfolio Optimization and Backtesting System

A comprehensive Python system for portfolio optimization, backtesting, and performance analysis using multiple optimization strategies.  Portfolio optimization is incorporated as a strategy during rebalancing.

## Features

- **Multiple Optimization Strategies**: Equal weight, Mean-variance (Sortino/Sharpe), Random weight, and Black-Litterman
- **Flexible Rebalancing**: Configurable rebalancing schedules (N_months, N_weeks, N_days)
- **Comprehensive Metrics**: Returns, drawdown, Sharpe ratio, Sortino ratio, Beta, VaR, CVaR, and more
- **Data Validation**: Robust data loading and validation with quality scoring
- **Transaction Costs**: Realistic transaction cost modeling
- **Visualization**: Portfolio performance charts and rolling metrics
- **Export Capabilities**: Results export to CSV, Excel, and JSON formats
- **Stock Basket Management**: Create, manage, and analyze collections of stocks/ETFs
- **Interactive Dashboard**: Multi-page Streamlit interface with basket configuration

### Limitations
- Survivor bias - does not easily have the capability of backtest universe selection (coming soon)

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

### Interactive Dashboard (Recommended)

The easiest way to get started is with the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

This will launch a multi-page web interface with:
- **Dashboard**: Main portfolio analysis and backtesting
- **Stock Baskets**: Create and manage collections of stocks/ETFs

### Run with Sample Data

You can also run the system with sample data from the command line:

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
   - Create a `data/price` directory
   - Add CSV files named `{TICKER}.csv` (e.g., `AAPL.csv`, `GOOGL.csv`)
   - Each CSV should have columns: `date`, `close`

2. Run the analysis:
```bash
python main.py --data-dir ./data/price --strategies equal_weight mean_variance
```

### Sample Jupyter notebook
- Example backtest and portfolio optimization strategy pipeline

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

### Upside Data Files

The system supports upside analysis data stored in CSV format in the `data/upside/` directory. These files contain analyst price targets and calculated upside potential for each security.

#### File Structure

Each ticker should have a CSV file named `{TICKER}.csv` in the `data/upside/` directory with the following structure:

```csv
close,price_target,upside
61.28,66.0,0.07702349869451695
60.42,72.0,0.19165839126117176
58.11,50.0,-0.13956289795215968
64.22,70.0,0.09000311429461229
...
```

#### Column Descriptions

- **close**: Current closing price of the security
- **price_target**: Analyst price target for the security
- **upside**: Calculated upside potential as a decimal (e.g., 0.077 = 7.7% upside)

#### Usage

Upside data is used by the Black-Litterman strategy to incorporate analyst views and price targets into portfolio optimization. The system automatically loads upside data when available and uses it to generate more informed portfolio allocations.



## Stock Basket Management

The system includes a powerful stock basket management feature that allows you to create, organize, and analyze collections of stocks and ETFs.

### What are Stock Baskets?

A stock basket is a named collection of stocks or ETFs that you can use for portfolio analysis. Baskets help you:
- Organize related securities (e.g., "Tech Stocks", "Dividend Stocks")
- Quickly switch between different investment themes
- Compare performance across different asset groups
- Share and reuse common portfolio configurations

### Creating and Managing Baskets

#### Using the Interactive Dashboard

1. **Navigate to Stock Baskets**: Click on "Stock Baskets" in the sidebar or page selector
2. **Create New Basket**: 
   - Click "Create New Basket" tab
   - Enter basket name and description
   - Select symbols from available data
   - Add optional tags for categorization
   - Click "Create Basket"

3. **Manage Existing Baskets**:
   - View all baskets in the "Manage Baskets" tab
   - Edit basket details, symbols, or tags
   - Delete unwanted baskets
   - Search and filter baskets by name or tags

#### Using the API

```python
from optfolio.data.basket_manager import StockBasketManager

# Initialize basket manager
basket_manager = StockBasketManager("data")

# Create a new basket
basket_id = basket_manager.create_basket(
    name="Tech Stocks",
    symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
    description="Major technology companies",
    tags=["tech", "large-cap"]
)

# Get basket information
basket = basket_manager.get_basket(basket_id)
print(f"Basket: {basket['name']}")
print(f"Symbols: {basket['symbols']}")

# Update basket
basket_manager.update_basket(
    basket_id,
    symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    tags=["tech", "large-cap", "faang"]
)

# Get all baskets
all_baskets = basket_manager.get_all_baskets()
for basket_id, basket in all_baskets.items():
    print(f"{basket['name']}: {len(basket['symbols'])} symbols")
```

### Using Baskets in Analysis

#### In the Dashboard

1. **Select Basket**: In the Dashboard page, use the "Stock Basket" selector in the sidebar
2. **Run Analysis**: The system will automatically load data for all symbols in the selected basket
3. **Compare Baskets**: Switch between different baskets to compare performance

#### In Code

```python
from optfolio.data.loader import DataLoader

# Initialize data loader
data_loader = DataLoader("data/price")

# Load data from a specific basket
prices = data_loader.load_prices_from_basket(basket_id)

# Get basket information
basket_info = data_loader.get_basket_info(basket_id)
print(f"Analyzing basket: {basket_info['name']}")
print(f"Symbols: {basket_info['symbols']}")

# Run backtesting with basket data
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory

backtester = Backtester(initial_capital=100000)
backtester.load_data(data_loader, tickers=prices.columns.tolist())

strategy = StrategyFactory.create('equal_weight')
results = backtester.run_backtest(strategy)
```

### Default Baskets

The system comes with several pre-configured baskets:

- **S&P 500 Tech**: Major technology stocks (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA)
- **Sector ETFs**: Sector SPDR ETFs for broad market exposure
- **Financial Services**: Major banks and financial companies
- **Healthcare & Biotech**: Healthcare and biotechnology companies

To create default baskets, click "Create Default Baskets" in the Stock Baskets page sidebar.

### Import/Export Baskets

#### Export Baskets

```python
# Export as JSON
json_data = basket_manager.export_basket(basket_id, "json")

# Export as CSV
csv_data = basket_manager.export_basket(basket_id, "csv")
```

#### Import Baskets

```python
# Import from JSON
basket_data = {
    "name": "My Custom Basket",
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "description": "Custom technology portfolio",
    "tags": ["custom", "tech"]
}
basket_id = basket_manager.import_basket(basket_data, "json")

# Import from CSV
csv_data = "symbol,basket_name\nAAPL,My Basket\nGOOGL,My Basket\nMSFT,My Basket"
basket_id = basket_manager.import_basket(csv_data, "csv")
```

### Basket Statistics

The system provides comprehensive statistics about your baskets:

```python
stats = basket_manager.get_basket_stats()
print(f"Total baskets: {stats['total_baskets']}")
print(f"Total symbols: {stats['total_symbols']}")
print(f"Unique symbols: {stats['unique_symbols']}")
print(f"Average symbols per basket: {stats['avg_symbols_per_basket']:.1f}")
print("Most common symbols:", stats['most_common_symbols'])
```

### Best Practices

1. **Naming Convention**: Use descriptive names like "Tech Large-Cap" or "Dividend Aristocrats"
2. **Tagging**: Add relevant tags for easy filtering and organization
3. **Symbol Validation**: The system automatically validates symbols against available data
4. **Regular Updates**: Keep baskets updated as your investment strategy evolves
5. **Documentation**: Use descriptions to explain the rationale behind each basket

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

# Run basket-specific tests
pytest optfolio/tests/test_basket_manager.py
pytest optfolio/tests/test_basket_integration.py

# Run all basket tests
pytest optfolio/tests/test_basket*.py
```

## Project Structure

```
optfolio/
├── data/                   # Data loading and validation
│   ├── loader.py          # CSV data loader with basket support
│   ├── validator.py       # Data validation
│   └── basket_manager.py  # Stock basket management
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
│   ├── test_basket_manager.py    # Basket manager tests
│   └── test_basket_integration.py # Basket integration tests
├── cli/               # Command line interface
├── config/            # Configuration management
├── pages/             # Streamlit multi-page app
│   ├── dashboard.py   # Main portfolio dashboard
│   └── stock_baskets.py # Basket configuration page
├── streamlit_app.py   # Main Streamlit entry point
└── data/              # Data directory
    ├── price/         # CSV price data files
    └── stock_baskets.json # Basket storage
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
