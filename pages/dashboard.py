"""Main Portfolio Dashboard Page for Streamlit App."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from optfolio.data.loader import DataLoader
from optfolio.data.validator import DataValidator
from optfolio.backtesting.engine import Backtester
from optfolio.strategies.base import StrategyFactory

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .strategy-comparison {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Define target symbols (fallback if no basket is selected)
DEFAULT_SYMBOLS = [
    'WSM', 'PAYX', 'BMI', 'BK', 'NDAQ', 'MSI', 'WMT', 'TJX', 'AIG', 'RJF', 
    'V', 'CTAS', 'TT', 'TRGP', 'JPM', 'GE', 'MCK', 'PH', 'LLY', 'COST', 
    'AVGO', 'NEE', 'AMAT', 'ADI', 'SHW', 'INTU', 'KLAC'
]

@st.cache_data
def load_data():
    """Load and cache the portfolio data."""
    try:
        data_dir = "data/price"
        data_loader = DataLoader(data_dir)
        
        # Load data for specific symbols only
        prices = data_loader.load_prices(tickers=DEFAULT_SYMBOLS)
        returns = data_loader.get_returns()
        
        # Filter returns to match the loaded prices
        returns = returns[prices.columns]
        
        return prices, returns, data_loader
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_data_from_basket(basket_id: str):
    """Load and cache the portfolio data from a specific basket."""
    try:
        data_dir = "data/price"
        data_loader = DataLoader(data_dir)
        
        # Load data from basket
        prices = data_loader.load_prices_from_basket(basket_id)
        returns = data_loader.get_returns()
        
        # Filter returns to match the loaded prices
        returns = returns[prices.columns]
        
        return prices, returns, data_loader
    except Exception as e:
        st.error(f"Error loading data from basket: {e}")
        return None, None, None

@st.cache_data
def run_backtests(prices, returns, _data_loader, start_date, end_date, rebalance_months):
    """Run backtests and cache results."""
    try:
        # Create strategies
        strategies = [
            StrategyFactory.create('equal_weight'),
            StrategyFactory.create('buy_and_hold', allocation_method="equal_weight"),
            StrategyFactory.create('buy_and_hold', allocation_method="market_cap"),
            StrategyFactory.create('mean_variance', objective="sortino_ratio"),
            StrategyFactory.create('mean_variance', objective="sharpe_ratio"),
            StrategyFactory.create('random_weight', distribution="dirichlet", seed=42),
            StrategyFactory.create('black_litterman', 
                                 prior_method="market_cap", view_method="momentum"),
            StrategyFactory.create('black_litterman', 
                                 prior_method="market_cap", view_method="upside")
        ]
        
        strategy_names = [
            "Equal Weight", "Buy and Hold (Equal Weight)", "Buy and Hold (Market Cap)", "Mean-Variance (Sortino)", "Mean-Variance (Sharpe)", 
            "Random Weight", "Black-Litterman (Momentum)", "Black-Litterman (Upside)"
        ]
        
        for i, strategy in enumerate(strategies):
            strategy.name = strategy_names[i]
            
            # Set minimum weight constraint for Black-Litterman strategies
            if 'black_litterman' in strategy.name.lower():
                strategy.min_weight = 0.01  # 1% minimum allocation
        
        # Set up backtesting
        backtester = Backtester(
            initial_capital=100000.0,
            risk_free_rate=0.02,
            transaction_costs=0.001
        )
        
        # Load data into backtester
        backtester.load_data(_data_loader, tickers=prices.columns.tolist())
        
        # Define rebalancing schedule based on configuration
        if rebalance_months == 0:
            # Buy and hold - no rebalancing
            rebalance_freq = None
        else:
            # Rebalance every N months
            rebalance_freq = {"months": rebalance_months, "weeks": 1, "days": 1}
        
        # Run backtests for each strategy individually to handle buy-and-hold correctly
        results = {}
        for strategy in strategies:
            # Buy-and-hold strategies should always use None rebalance_freq
            if 'buy and hold' in strategy.name.lower():
                strategy_rebalance_freq = None
            else:
                strategy_rebalance_freq = rebalance_freq
            
            # Run backtest for this strategy
            strategy_result = backtester.run_backtest(
                strategy=strategy,
                rebalance_freq=strategy_rebalance_freq,
                start_date=start_date,
                end_date=end_date
            )
            results[strategy.name] = strategy_result
        
        return results, strategies
        
    except Exception as e:
        st.error(f"Error running backtests: {e}")
        return None, None

def create_strategy_comparison_table(results):
    """Create a formatted strategy comparison table."""
    strategy_results = []
    
    for strategy_name, result in results.items():
        metrics = result['performance_metrics']
        summary = result['summary']
        significance = result.get('significance', {})
        
        # Format p-value with appropriate precision
        p_value = significance.get('p_value', np.nan)
        if np.isnan(p_value):
            p_value_str = "N/A"
        elif p_value < 0.001:
            p_value_str = "< 0.001"
        else:
            p_value_str = f"{p_value:.3f}"
        
        strategy_results.append({
            'Strategy': strategy_name,
            'Total Return (%)': f"{metrics.get('total_return', 0):.2f}",
            'Annualized Return (%)': f"{metrics.get('annualized_return', 0):.2f}",
            'Volatility (%)': f"{metrics.get('volatility', 0):.2f}",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
            'Sortino Ratio': f"{metrics.get('sortino_ratio', 0):.3f}",
            'Max Drawdown (%)': f"{metrics.get('max_drawdown', 0):.2f}",
            'P-Value (t-test)': p_value_str,
            'Rebalances': summary.get('num_rebalances', 0),
            'Transactions': summary.get('num_transactions', 0),
            'Transaction Costs ($)': f"{summary.get('total_transaction_costs', 0):.2f}"
        })
    
    return pd.DataFrame(strategy_results)

def plot_portfolio_values(results, selected_strategy):
    """Create portfolio value over time chart."""
    if selected_strategy not in results:
        return None
    
    result = results[selected_strategy]
    if 'portfolio_values' not in result or len(result['portfolio_values']) == 0:
        return None
    
    portfolio_series = result['portfolio_values']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_series.index,
        y=portfolio_series.values,
        mode='lines',
        name=selected_strategy,
        line=dict(width=3, color='#1f77b4')
    ))
    
    fig.update_layout(
        title=f"Portfolio Value Over Time - {selected_strategy}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig

def plot_all_strategies_comparison(results):
    """Create portfolio value over time chart for all strategies."""
    if not results:
        return None
    
    fig = go.Figure()
    
    # Define colors for different strategies
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (strategy_name, result) in enumerate(results.items()):
        if 'portfolio_values' not in result or len(result['portfolio_values']) == 0:
            continue
            
        portfolio_series = result['portfolio_values']
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=portfolio_series.index,
            y=portfolio_series.values,
            mode='lines',
            name=strategy_name,
            line=dict(width=2, color=color),
            hovertemplate=f'<b>{strategy_name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Portfolio Value Comparison - All Strategies",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig

def main():
    """Main function for the dashboard page."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Portfolio Optimization Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Basket selection
    st.sidebar.subheader("📊 Stock Basket")
    try:
        data_loader = DataLoader("data/price")
        all_baskets = data_loader.get_all_baskets()
        
        if all_baskets:
            basket_options = {basket_id: basket["name"] for basket_id, basket in all_baskets.items()}
            basket_options["default"] = "Default Symbols (No Basket)"
            
            selected_basket = st.sidebar.selectbox(
                "Select Stock Basket",
                options=list(basket_options.keys()),
                format_func=lambda x: basket_options[x],
                index=0
            )
            
            if selected_basket == "default":
                use_basket = False
                basket_info = None
            else:
                use_basket = True
                basket_info = data_loader.get_basket_info(selected_basket)
                st.sidebar.write(f"**Basket:** {basket_info['name']}")
                st.sidebar.write(f"**Symbols:** {len(basket_info['symbols'])}")
                st.sidebar.write(f"**Description:** {basket_info.get('description', 'No description')}")
        else:
            use_basket = False
            basket_info = None
            st.sidebar.info("No baskets available. Using default symbols.")
    except Exception as e:
        st.sidebar.error(f"Error loading baskets: {e}")
        use_basket = False
        basket_info = None
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=pd.to_datetime('2023-01-01').date(),
        min_value=pd.to_datetime('2015-01-01').date(),
        max_value=pd.to_datetime('2025-12-31').date()
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=pd.to_datetime('2025-08-31').date(),
        min_value=pd.to_datetime('2015-01-01').date(),
        max_value=pd.to_datetime('2025-12-31').date()
    )
    
    # Rebalancing configuration
    st.sidebar.subheader("🔄 Rebalancing")
    rebalance_months = st.sidebar.selectbox(
        "Rebalance Period (months)",
        options=[0, 1, 2, 3, 6, 12],
        index=3,  # Default to 3 months
        help="0 = Buy and Hold (no rebalancing), 1-12 = Rebalance every N months"
    )
    
    if rebalance_months == 0:
        st.sidebar.info("📌 **Buy and Hold Strategy**: No rebalancing will occur")
    else:
        st.sidebar.info(f"🔄 **Rebalancing**: Every {rebalance_months} month{'s' if rebalance_months > 1 else ''}")
    
    # Load data
    with st.spinner("Loading data..."):
        if use_basket and basket_info:
            prices, returns, data_loader = load_data_from_basket(selected_basket)
            symbols_info = f"Basket: {basket_info['name']} ({len(basket_info['symbols'])} symbols)"
        else:
            prices, returns, data_loader = load_data()
            symbols_info = f"Default symbols ({len(DEFAULT_SYMBOLS)} symbols)"
    
    if prices is None or returns is None:
        st.error("Failed to load data. Please check that the data directory exists and contains the required CSV files.")
        return
    
    # Data summary
    st.sidebar.subheader("📊 Data Summary")
    st.sidebar.metric("Symbols Loaded", len(prices.columns))
    st.sidebar.metric("Date Range", f"{prices.index.min().date()} to {prices.index.max().date()}")
    st.sidebar.metric("Total Observations", len(prices))
    
    # Run backtests
    with st.spinner("Running backtests..."):
        results, strategies = run_backtests(prices, returns, data_loader, str(start_date), str(end_date), rebalance_months)
    
    if results is None:
        st.error("Failed to run backtests. Please check the configuration.")
        return
    
    # Strategy comparison table
    st.header("📋 Strategy Performance Comparison")
    
    comparison_df = create_strategy_comparison_table(results)
    
    # Display table with better formatting
    st.markdown('<div class="strategy-comparison">', unsafe_allow_html=True)
    st.dataframe(
        comparison_df,
        width='stretch',
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Portfolio value comparison chart for all strategies
    st.header("📈 Portfolio Value Comparison")
    all_strategies_fig = plot_all_strategies_comparison(results)
    if all_strategies_fig:
        st.plotly_chart(all_strategies_fig, width='stretch')
    else:
        st.warning("Portfolio value data not available for comparison.")
    
    # Strategy selection
    st.header("📊 Strategy Analysis")
    
    selected_strategy = st.selectbox(
        "Select a strategy for detailed analysis:",
        options=list(results.keys()),
        index=0
    )
    
    # Display metrics for selected strategy
    if selected_strategy in results:
        result = results[selected_strategy]
        metrics = result['performance_metrics']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.2%}",
                delta=f"{metrics.get('annualized_return', 0):.2%} (annualized)"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.3f}",
                delta=f"{metrics.get('sortino_ratio', 0):.3f} (Sortino)"
            )
        
        with col3:
            st.metric(
                "Volatility",
                f"{metrics.get('volatility', 0):.2%}",
                delta=f"{metrics.get('max_drawdown', 0):.2%} (max drawdown)"
            )
        
        with col4:
            summary = result['summary']
            st.metric(
                "Rebalances",
                summary.get('num_rebalances', 0),
                delta=f"{summary.get('num_transactions', 0)} (trades)"
            )
        
        with col5:
            summary = result['summary']
            st.metric(
                "Transaction Costs",
                f"${summary.get('total_transaction_costs', 0):.2f}",
                delta=f"{summary.get('num_transactions', 0)} trades"
            )
    
    # Portfolio value over time
    portfolio_fig = plot_portfolio_values(results, selected_strategy)
    if portfolio_fig:
        st.plotly_chart(portfolio_fig, width='stretch')
    else:
        st.warning("Portfolio value data not available for the selected strategy.")
    
    # Footer
    st.markdown("---")
    rebalance_info = "Buy & Hold" if rebalance_months == 0 else f"Every {rebalance_months} month{'s' if rebalance_months > 1 else ''}"
    st.markdown(
        "**Portfolio Optimization Dashboard** | Built with Streamlit and Plotly | "
        f"Data: {symbols_info} | "
        f"Period: {start_date} to {end_date} | "
        f"Rebalancing: {rebalance_info}"
    )

if __name__ == "__main__":
    main()

