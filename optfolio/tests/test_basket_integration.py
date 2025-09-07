"""Tests for basket integration with DataLoader."""

import pytest
import tempfile
import shutil
from pathlib import Path

from optfolio.data.loader import DataLoader
from optfolio.data.basket_manager import StockBasketManager


class TestBasketIntegration:
    """Test cases for basket integration with DataLoader."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_price_files(self, temp_dir):
        """Create sample price CSV files for testing."""
        price_dir = temp_dir / "price"
        price_dir.mkdir()
        
        # Create sample CSV files with more realistic data
        sample_data = {
            "AAPL": [
                ("2023-01-01", 150.0), ("2023-01-02", 151.0), ("2023-01-03", 152.0),
                ("2023-01-04", 149.0), ("2023-01-05", 153.0)
            ],
            "GOOGL": [
                ("2023-01-01", 2800.0), ("2023-01-02", 2810.0), ("2023-01-03", 2820.0),
                ("2023-01-04", 2790.0), ("2023-01-05", 2830.0)
            ],
            "MSFT": [
                ("2023-01-01", 300.0), ("2023-01-02", 301.0), ("2023-01-03", 302.0),
                ("2023-01-04", 299.0), ("2023-01-05", 303.0)
            ],
            "TSLA": [
                ("2023-01-01", 200.0), ("2023-01-02", 201.0), ("2023-01-03", 202.0),
                ("2023-01-04", 199.0), ("2023-01-05", 203.0)
            ]
        }
        
        for symbol, data in sample_data.items():
            csv_file = price_dir / f"{symbol}.csv"
            with open(csv_file, 'w') as f:
                f.write("date,close\n")
                for date, price in data:
                    f.write(f"{date},{price}\n")
        
        return sample_data
    
    @pytest.fixture
    def data_loader(self, temp_dir):
        """Create a DataLoader instance for testing."""
        return DataLoader(temp_dir / "price")
    
    def test_data_loader_basket_manager_initialization(self, data_loader):
        """Test that DataLoader initializes basket manager correctly."""
        assert hasattr(data_loader, 'basket_manager')
        assert isinstance(data_loader.basket_manager, StockBasketManager)
    
    def test_load_prices_from_basket(self, data_loader, sample_price_files):
        """Test loading prices from a specific basket."""
        # Create a basket
        basket_id = data_loader.basket_manager.create_basket(
            "Tech Stocks",
            ["AAPL", "GOOGL", "MSFT"],
            "Technology companies"
        )
        
        # Load prices from basket
        prices = data_loader.load_prices_from_basket(basket_id)
        
        # Verify the data
        assert prices is not None
        assert len(prices.columns) == 3
        assert set(prices.columns) == {"AAPL", "GOOGL", "MSFT"}
        assert len(prices) == 5  # 5 days of data
        
        # Check that data is properly loaded
        assert prices.index[0].strftime("%Y-%m-%d") == "2023-01-01"
        assert prices.index[-1].strftime("%Y-%m-%d") == "2023-01-05"
        assert prices.loc["2023-01-01", "AAPL"] == 150.0
        assert prices.loc["2023-01-01", "GOOGL"] == 2800.0
        assert prices.loc["2023-01-01", "MSFT"] == 300.0
    
    def test_load_prices_from_basket_not_found(self, data_loader):
        """Test loading prices from non-existent basket."""
        with pytest.raises(ValueError, match="Basket non-existent-id not found"):
            data_loader.load_prices_from_basket("non-existent-id")
    
    def test_load_prices_from_basket_empty_symbols(self, data_loader, sample_price_files):
        """Test loading prices from basket with no symbols."""
        # Try to create empty basket - this should fail
        with pytest.raises(ValueError, match="No valid symbols found"):
            data_loader.basket_manager.create_basket("Empty Basket", [])
    
    def test_get_basket_info(self, data_loader, sample_price_files):
        """Test getting basket information through DataLoader."""
        # Create a basket
        basket_id = data_loader.basket_manager.create_basket(
            "Test Basket",
            ["AAPL", "GOOGL"],
            "Test description",
            ["test", "tech"]
        )
        
        # Get basket info
        basket_info = data_loader.get_basket_info(basket_id)
        
        assert basket_info is not None
        assert basket_info["name"] == "Test Basket"
        assert basket_info["symbols"] == ["AAPL", "GOOGL"]
        assert basket_info["description"] == "Test description"
        assert basket_info["tags"] == ["test", "tech"]
    
    def test_get_basket_info_not_found(self, data_loader):
        """Test getting info for non-existent basket."""
        basket_info = data_loader.get_basket_info("non-existent-id")
        assert basket_info is None
    
    def test_get_all_baskets(self, data_loader, sample_price_files):
        """Test getting all baskets through DataLoader."""
        # Initially no baskets
        baskets = data_loader.get_all_baskets()
        assert len(baskets) == 0
        
        # Create some baskets
        basket1_id = data_loader.basket_manager.create_basket("Basket 1", ["AAPL"])
        basket2_id = data_loader.basket_manager.create_basket("Basket 2", ["GOOGL"])
        
        # Get all baskets
        baskets = data_loader.get_all_baskets()
        assert len(baskets) == 2
        assert basket1_id in baskets
        assert basket2_id in baskets
    
    def test_get_basket_stats(self, data_loader, sample_price_files):
        """Test getting basket statistics through DataLoader."""
        # Initially no baskets
        stats = data_loader.get_basket_stats()
        assert stats["total_baskets"] == 0
        
        # Create some baskets
        data_loader.basket_manager.create_basket("Basket 1", ["AAPL", "GOOGL"])
        data_loader.basket_manager.create_basket("Basket 2", ["MSFT", "TSLA"])
        
        # Get stats
        stats = data_loader.get_basket_stats()
        assert stats["total_baskets"] == 2
        assert stats["total_symbols"] == 4
        assert stats["unique_symbols"] == 4
    
    def test_basket_data_consistency(self, data_loader, sample_price_files):
        """Test that basket data is consistent with available price data."""
        # Create basket with all available symbols
        basket_id = data_loader.basket_manager.create_basket(
            "All Stocks",
            ["AAPL", "GOOGL", "MSFT", "TSLA"]
        )
        
        # Load prices from basket
        prices = data_loader.load_prices_from_basket(basket_id)
        
        # Verify all symbols are present
        assert set(prices.columns) == {"AAPL", "GOOGL", "MSFT", "TSLA"}
        
        # Verify data integrity
        for symbol in prices.columns:
            assert not prices[symbol].isnull().any()
            assert (prices[symbol] > 0).all()
    
    def test_basket_with_invalid_symbols(self, data_loader, sample_price_files):
        """Test basket creation with some invalid symbols."""
        # Create basket with mix of valid and invalid symbols
        basket_id = data_loader.basket_manager.create_basket(
            "Mixed Basket",
            ["AAPL", "INVALID_SYMBOL", "GOOGL", "MISSING_SYMBOL"]
        )
        
        # Only valid symbols should be included
        basket = data_loader.basket_manager.get_basket(basket_id)
        assert set(basket["symbols"]) == {"AAPL", "GOOGL"}
        
        # Load prices should work with only valid symbols
        prices = data_loader.load_prices_from_basket(basket_id)
        assert set(prices.columns) == {"AAPL", "GOOGL"}
    
    def test_basket_returns_calculation(self, data_loader, sample_price_files):
        """Test that returns calculation works with basket data."""
        # Create basket
        basket_id = data_loader.basket_manager.create_basket("Test Basket", ["AAPL", "GOOGL"])
        
        # Load prices from basket
        prices = data_loader.load_prices_from_basket(basket_id)
        
        # Calculate returns
        returns = data_loader.get_returns()
        
        # Verify returns are calculated correctly
        assert returns is not None
        assert set(returns.columns) == {"AAPL", "GOOGL"}
        assert len(returns) == len(prices)  # Returns has same length as prices (first row is NaN)
        
        # Check that returns are reasonable (not all NaN)
        assert not returns.isnull().all().all()
        
        # Check that first row is NaN (no return for first day)
        assert returns.iloc[0].isnull().all()
    
    def test_multiple_baskets_isolation(self, data_loader, sample_price_files):
        """Test that multiple baskets don't interfere with each other."""
        # Create two different baskets
        basket1_id = data_loader.basket_manager.create_basket("Basket 1", ["AAPL", "GOOGL"])
        basket2_id = data_loader.basket_manager.create_basket("Basket 2", ["MSFT", "TSLA"])
        
        # Load prices from each basket
        prices1 = data_loader.load_prices_from_basket(basket1_id)
        prices2 = data_loader.load_prices_from_basket(basket2_id)
        
        # Verify they have different symbols
        assert set(prices1.columns) == {"AAPL", "GOOGL"}
        assert set(prices2.columns) == {"MSFT", "TSLA"}
        
        # Verify data is correct for each
        assert prices1.loc["2023-01-01", "AAPL"] == 150.0
        assert prices2.loc["2023-01-01", "MSFT"] == 300.0
    
    def test_basket_update_and_reload(self, data_loader, sample_price_files):
        """Test updating a basket and reloading data."""
        # Create initial basket
        basket_id = data_loader.basket_manager.create_basket("Initial Basket", ["AAPL"])
        
        # Load initial prices
        prices_initial = data_loader.load_prices_from_basket(basket_id)
        assert set(prices_initial.columns) == {"AAPL"}
        
        # Update basket to include more symbols
        data_loader.basket_manager.update_basket(basket_id, symbols=["AAPL", "GOOGL", "MSFT"])
        
        # Reload prices
        prices_updated = data_loader.load_prices_from_basket(basket_id)
        assert set(prices_updated.columns) == {"AAPL", "GOOGL", "MSFT"}
        
        # Verify original data is still there
        assert prices_updated.loc["2023-01-01", "AAPL"] == 150.0
        assert prices_updated.loc["2023-01-01", "GOOGL"] == 2800.0
        assert prices_updated.loc["2023-01-01", "MSFT"] == 300.0
