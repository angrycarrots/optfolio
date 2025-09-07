"""Stock Basket Configuration Page for Streamlit App."""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Optional
from pathlib import Path

# Import our modules
from optfolio.data.basket_manager import StockBasketManager
from optfolio.data.loader import DataLoader

# Page configuration
st.set_page_config(
    page_title="Stock Basket Configuration",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .basket-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .symbol-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        margin: 0.125rem;
        display: inline-block;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_basket_manager():
    """Load and cache the basket manager."""
    try:
        return StockBasketManager("data")
    except Exception as e:
        st.error(f"Error loading basket manager: {e}")
        return None

@st.cache_data
def load_data_loader():
    """Load and cache the data loader."""
    try:
        return DataLoader("data/price")
    except Exception as e:
        st.error(f"Error loading data loader: {e}")
        return None

def display_basket_card(basket: Dict, basket_id: str):
    """Display a basket as a card."""
    with st.container():
        st.markdown(f'<div class="basket-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(basket["name"])
            if basket.get("description"):
                st.write(basket["description"])
            
            # Display symbols as tags
            symbols = basket.get("symbols", [])
            if symbols:
                symbol_tags = " ".join([f'<span class="symbol-tag">{symbol}</span>' for symbol in symbols[:10]])
                if len(symbols) > 10:
                    symbol_tags += f' <span class="symbol-tag">+{len(symbols) - 10} more</span>'
                st.markdown(symbol_tags, unsafe_allow_html=True)
            
            # Display tags
            tags = basket.get("tags", [])
            if tags:
                st.write(f"**Tags:** {', '.join(tags)}")
        
        with col2:
            st.metric("Symbols", len(symbols))
            st.metric("Created", basket.get("created_date", "Unknown")[:10])
        
        with col3:
            if st.button("Edit", key=f"edit_{basket_id}"):
                st.session_state[f"edit_basket_{basket_id}"] = True
            
            if st.button("Delete", key=f"delete_{basket_id}"):
                st.session_state[f"delete_basket_{basket_id}"] = True
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_basket_form():
    """Create a form for creating/editing baskets."""
    st.subheader("Create New Basket")
    
    with st.form("create_basket_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Basket Name", placeholder="e.g., Tech Stocks")
            description = st.text_area("Description", placeholder="Optional description")
        
        with col2:
            # Get available symbols
            basket_manager = load_basket_manager()
            if basket_manager:
                available_symbols = basket_manager.get_available_symbols()
                selected_symbols = st.multiselect(
                    "Select Symbols",
                    options=available_symbols,
                    default=[],
                    help="Search and select symbols from available data"
                )
            else:
                st.error("Cannot load available symbols")
                selected_symbols = []
        
        tags = st.text_input("Tags (comma-separated)", placeholder="e.g., tech, large-cap, growth")
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        submitted = st.form_submit_button("Create Basket")
        
        if submitted:
            if not name:
                st.error("Basket name is required")
            elif not selected_symbols:
                st.error("At least one symbol must be selected")
            else:
                try:
                    basket_manager = load_basket_manager()
                    if basket_manager:
                        basket_id = basket_manager.create_basket(
                            name=name,
                            symbols=selected_symbols,
                            description=description or None,
                            tags=tags_list or None
                        )
                        st.success(f"Basket '{name}' created successfully!")
                        st.rerun()
                except ValueError as e:
                    st.error(f"Error creating basket: {e}")

def edit_basket_form(basket_id: str, basket: Dict):
    """Create a form for editing an existing basket."""
    st.subheader(f"Edit Basket: {basket['name']}")
    
    with st.form(f"edit_basket_form_{basket_id}"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Basket Name", value=basket["name"])
            description = st.text_area("Description", value=basket.get("description", ""))
        
        with col2:
            # Get available symbols
            basket_manager = load_basket_manager()
            if basket_manager:
                available_symbols = basket_manager.get_available_symbols()
                selected_symbols = st.multiselect(
                    "Select Symbols",
                    options=available_symbols,
                    default=basket.get("symbols", []),
                    help="Search and select symbols from available data"
                )
            else:
                st.error("Cannot load available symbols")
                selected_symbols = basket.get("symbols", [])
        
        tags = st.text_input("Tags (comma-separated)", value=", ".join(basket.get("tags", [])))
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Save Changes")
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state[f"edit_basket_{basket_id}"] = False
                st.rerun()
        
        if submitted:
            try:
                basket_manager = load_basket_manager()
                if basket_manager:
                    basket_manager.update_basket(
                        basket_id,
                        name=name,
                        symbols=selected_symbols,
                        description=description or None,
                        tags=tags_list or None
                    )
                    st.success(f"Basket '{name}' updated successfully!")
                    st.session_state[f"edit_basket_{basket_id}"] = False
                    st.rerun()
            except ValueError as e:
                st.error(f"Error updating basket: {e}")

def delete_basket_confirmation(basket_id: str, basket: Dict):
    """Show confirmation dialog for deleting a basket."""
    st.warning(f"Are you sure you want to delete the basket '{basket['name']}'?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", key=f"confirm_delete_{basket_id}"):
            try:
                basket_manager = load_basket_manager()
                if basket_manager:
                    basket_manager.delete_basket(basket_id)
                    st.success(f"Basket '{basket['name']}' deleted successfully!")
                    st.session_state[f"delete_basket_{basket_id}"] = False
                    st.rerun()
            except Exception as e:
                st.error(f"Error deleting basket: {e}")
    
    with col2:
        if st.button("Cancel", key=f"cancel_delete_{basket_id}"):
            st.session_state[f"delete_basket_{basket_id}"] = False
            st.rerun()

def display_basket_stats():
    """Display basket statistics."""
    basket_manager = load_basket_manager()
    if not basket_manager:
        return
    
    stats = basket_manager.get_basket_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Baskets", stats["total_baskets"])
    
    with col2:
        st.metric("Total Symbols", stats["total_symbols"])
    
    with col3:
        st.metric("Unique Symbols", stats["unique_symbols"])
    
    with col4:
        st.metric("Avg Symbols/Basket", f"{stats['avg_symbols_per_basket']:.1f}")
    
    # Most common symbols
    if stats["most_common_symbols"]:
        st.subheader("Most Common Symbols")
        common_df = pd.DataFrame(stats["most_common_symbols"], columns=["Symbol", "Count"])
        st.dataframe(common_df, hide_index=True)

def main():
    """Main function for the stock baskets page."""
    st.title("📊 Stock Basket Configuration")
    st.markdown("Create and manage collections of stocks and ETFs for portfolio analysis.")
    
    # Load managers
    basket_manager = load_basket_manager()
    if not basket_manager:
        st.error("Failed to load basket manager. Please check the data directory.")
        return
    
    # Sidebar for actions
    st.sidebar.header("Actions")
    
    if st.sidebar.button("Create Default Baskets"):
        try:
            created_baskets = basket_manager.create_default_baskets()
            if created_baskets:
                st.success(f"Created {len(created_baskets)} default baskets!")
                st.rerun()
            else:
                st.info("Default baskets already exist or could not be created.")
        except Exception as e:
            st.error(f"Error creating default baskets: {e}")
    
    # Search and filter
    st.sidebar.subheader("Search & Filter")
    search_query = st.sidebar.text_input("Search baskets", placeholder="Search by name or description")
    
    # Get all baskets
    all_baskets = basket_manager.get_all_baskets()
    
    # Filter baskets based on search
    filtered_baskets = {}
    if search_query:
        for basket_id, basket in all_baskets.items():
            if (search_query.lower() in basket["name"].lower() or 
                search_query.lower() in basket.get("description", "").lower()):
                filtered_baskets[basket_id] = basket
    else:
        filtered_baskets = all_baskets
    
    # Display statistics
    if all_baskets:
        display_basket_stats()
        st.markdown("---")
    
    # Main content area
    tab1, tab2 = st.tabs(["Manage Baskets", "Create New Basket"])
    
    with tab1:
        if not filtered_baskets:
            if search_query:
                st.info(f"No baskets found matching '{search_query}'")
            else:
                st.info("No baskets created yet. Create your first basket in the 'Create New Basket' tab.")
        else:
            st.subheader(f"Baskets ({len(filtered_baskets)})")
            
            # Check for edit/delete actions
            for basket_id, basket in filtered_baskets.items():
                if st.session_state.get(f"edit_basket_{basket_id}", False):
                    edit_basket_form(basket_id, basket)
                elif st.session_state.get(f"delete_basket_{basket_id}", False):
                    delete_basket_confirmation(basket_id, basket)
                else:
                    display_basket_card(basket, basket_id)
    
    with tab2:
        create_basket_form()
    
    # Export/Import section
    st.markdown("---")
    st.subheader("Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Basket**")
        if all_baskets:
            basket_names = {basket_id: basket["name"] for basket_id, basket in all_baskets.items()}
            selected_basket_id = st.selectbox("Select basket to export", options=list(basket_names.keys()), 
                                            format_func=lambda x: basket_names[x])
            
            export_format = st.selectbox("Export format", ["json", "csv"])
            
            if st.button("Export"):
                try:
                    exported_data = basket_manager.export_basket(selected_basket_id, export_format)
                    if export_format == "json":
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(exported_data, indent=2),
                            file_name=f"{basket_names[selected_basket_id]}.json",
                            mime="application/json"
                        )
                    else:  # CSV
                        st.download_button(
                            label="Download CSV",
                            data=exported_data,
                            file_name=f"{basket_names[selected_basket_id]}.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error exporting basket: {e}")
        else:
            st.info("No baskets available for export")
    
    with col2:
        st.write("**Import Basket**")
        uploaded_file = st.file_uploader("Choose a file", type=['json', 'csv'])
        
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read().decode('utf-8')
                file_format = uploaded_file.name.split('.')[-1].lower()
                
                if st.button("Import"):
                    basket_id = basket_manager.import_basket(file_content, file_format)
                    st.success(f"Basket imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing basket: {e}")

if __name__ == "__main__":
    main()
