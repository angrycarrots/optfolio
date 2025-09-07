#!/usr/bin/env python3
"""
Streamlit Portfolio Optimization Dashboard

Interactive dashboard for portfolio optimization and backtesting with enhanced visualizations.
This is the main entry point that redirects to the multi-page app.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Redirect to dashboard page
st.markdown("""
# 📊 Portfolio Optimization Dashboard

Welcome to the Portfolio Optimization Dashboard! This application has been updated with a multi-page structure.

**Available Pages:**
- **Dashboard**: Main portfolio analysis and backtesting
- **Stock Baskets**: Create and manage stock/ETF collections

Please navigate to the **Dashboard** page to access the main portfolio optimization features.

---

*Note: The main dashboard functionality has been moved to the Dashboard page for better organization.*
""")

# Add navigation instructions
st.info("""
**Navigation**: Use the sidebar or the page selector at the top to navigate between:
- 🏠 **Dashboard** - Main portfolio analysis
- 📊 **Stock Baskets** - Manage stock collections
""")

# Optional: Add a button to redirect to dashboard
if st.button("Go to Dashboard", type="primary"):
    st.switch_page("pages/dashboard.py")

if __name__ == "__main__":
    pass
