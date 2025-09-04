#!/usr/bin/env python3
"""
Streamlit Portfolio Optimization Dashboard

Interactive dashboard for portfolio optimization and backtesting with enhanced visualizations.
"""

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

# Define target symbols (same as example.py)
TARGET_SYMBOLS = [
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
        prices = data_loader.load_prices(tickers=TARGET_SYMBOLS)
        returns = data_loader.get_returns()
        
        # Filter returns to match the loaded prices
        returns = returns[prices.columns]
        
        return prices, returns, data_loader
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def run_backtests(prices, returns, _data_loader, start_date, end_date, rebalance_months):
    """Run backtests and cache results."""
    try:
        # Create strategies
        strategies = [
            StrategyFactory.create('equal_weight'),
            StrategyFactory.create('mean_variance', objective="sortino_ratio"),
            StrategyFactory.create('mean_variance', objective="sharpe_ratio"),
            StrategyFactory.create('random_weight', distribution="dirichlet", seed=42),
            StrategyFactory.create('black_litterman', 
                                 prior_method="market_cap", view_method="momentum"),
            StrategyFactory.create('black_litterman', 
                                 prior_method="market_cap", view_method="upside")
        ]
        
        strategy_names = [
            "Equal Weight", "Mean-Variance (Sortino)", "Mean-Variance (Sharpe)", 
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
        
        # Run backtests for all strategies
        results = backtester.run_multiple_backtests(
            strategies=strategies,
            rebalance_freq=rebalance_freq,
            start_date=start_date,
            end_date=end_date
        )
        
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

def plot_asset_allocation(results, selected_strategy):
    """Create stacked asset allocation chart over time."""
    if selected_strategy not in results:
        return None
    
    result = results[selected_strategy]
    if 'weight_history' not in result:
        return None
    
    weight_history = result['weight_history']
    if weight_history.empty:
        return None
    
    # Convert to DataFrame if it's not already
    if isinstance(weight_history, dict):
        weight_df = pd.DataFrame(weight_history)
    else:
        weight_df = weight_history
    
    # Ensure all weights sum to 1.0 (100%) for each time period
    weight_df = weight_df.div(weight_df.sum(axis=1), axis=0)
    
    # Get all assets, sorted by average weight (descending)
    avg_weights = weight_df.mean().sort_values(ascending=False)
    all_assets = avg_weights.index.tolist()
    
    fig = go.Figure()
    
    # Use a larger color palette to handle many assets
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Pastel2
    
    for i, asset in enumerate(all_assets):
        if asset in weight_df.columns:
            fig.add_trace(go.Scatter(
                x=weight_df.index,
                y=weight_df[asset],
                mode='lines',
                name=asset,
                stackgroup='one',
                line=dict(width=0),
                fillcolor=colors[i % len(colors)],
                hovertemplate=f'<b>{asset}</b><br>' +
                             'Date: %{x}<br>' +
                             'Weight: %{y:.1%}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Asset Allocation Over Time - {selected_strategy}",
        xaxis_title="Date",
        yaxis_title="Portfolio Weight",
        hovermode='x unified',
        template="plotly_white",
        height=500,
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1]  # Ensure y-axis goes from 0% to 100%
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def plot_rolling_sharpe(results, selected_strategy):
    """Create rolling Sharpe ratio chart."""
    if selected_strategy not in results:
        return None
    
    result = results[selected_strategy]
    if 'rolling_metrics' not in result or result['rolling_metrics'].empty:
        return None
    
    rolling_metrics = result['rolling_metrics']
    if 'rolling_sharpe' not in rolling_metrics.columns:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_metrics.index,
        y=rolling_metrics['rolling_sharpe'],
        mode='lines',
        name='Rolling Sharpe Ratio',
        line=dict(width=2, color='#ff7f0e')
    ))
    
    # Add horizontal line at Sharpe = 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Rolling Sharpe Ratio - {selected_strategy}",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode='x unified',
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_final_weights(results, selected_strategy):
    """Create final portfolio weights bar chart."""
    if selected_strategy not in results:
        return None
    
    result = results[selected_strategy]
    if 'last_weights' not in result:
        return None
    
    final_weights = result['last_weights']
    
    # Sort by weight
    sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
    assets, weights = zip(*sorted_weights)
    
    # Create horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(assets),
        x=list(weights),
        orientation='h',
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title=f"Final Portfolio Weights - {selected_strategy}",
        xaxis_title="Portfolio Weight",
        yaxis_title="Asset",
        template="plotly_white",
        height=max(400, len(assets) * 25),
        xaxis=dict(tickformat='.1%')
    )
    
    return fig

def main():
    """Main Streamlit app."""
    
    # Header
    # st.markdown('<h1 class="main-header">📊 Portfolio Optimization Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=pd.to_datetime('2017-01-01').date(),
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
    
    # Symbol information
    st.sidebar.subheader("📈 Target Symbols")
    st.sidebar.write(f"**{len(TARGET_SYMBOLS)} symbols selected:**")
    st.sidebar.write(", ".join(TARGET_SYMBOLS))
    
    # Load data
    with st.spinner("Loading data..."):
        prices, returns, data_loader = load_data()
    
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
        
        col1, col2, col3, col4 = st.columns(4)
        
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
                "Transactions",
                summary.get('num_transactions', 0),
                delta=f"${summary.get('total_transaction_costs', 0):.2f} (costs)"
            )
    
    # Charts
    # st.header("📈 Visualizations")
    
    # Portfolio value over time
    portfolio_fig = plot_portfolio_values(results, selected_strategy)
    if portfolio_fig:
        st.plotly_chart(portfolio_fig, width='stretch')
    else:
        st.warning("Portfolio value data not available for the selected strategy.")
    
    # Two-column layout for remaining charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset allocation over time
        allocation_fig = plot_asset_allocation(results, selected_strategy)
        if allocation_fig:
            st.plotly_chart(allocation_fig, width='stretch')
        else:
            st.warning("Asset allocation data not available for the selected strategy.")
    
    with col2:
        # Rolling Sharpe ratio
        sharpe_fig = plot_rolling_sharpe(results, selected_strategy)
        if sharpe_fig:
            st.plotly_chart(sharpe_fig, width='stretch')
        else:
            st.warning("Rolling Sharpe ratio data not available for the selected strategy.")
    
    # Final portfolio weights
    weights_fig = plot_final_weights(results, selected_strategy)
    if weights_fig:
        st.plotly_chart(weights_fig, width='stretch')
    else:
        st.warning("Final portfolio weights not available for the selected strategy.")
    
    # Footer
    st.markdown("---")
    rebalance_info = "Buy & Hold" if rebalance_months == 0 else f"Every {rebalance_months} month{'s' if rebalance_months > 1 else ''}"
    st.markdown(
        "**Portfolio Optimization Dashboard** | Built with Streamlit and Plotly | "
        f"Data: {len(TARGET_SYMBOLS)} symbols | "
        f"Period: {start_date} to {end_date} | "
        f"Rebalancing: {rebalance_info}"
    )

if __name__ == "__main__":
    main()
