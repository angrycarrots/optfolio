"""Tests for Streamlit basket functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optfolio.data.basket_manager import StockBasketManager
from optfolio.data.loader import DataLoader


class TestStreamlitBaskets:
    """Test cases for Streamlit basket functionality."""
    
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
        
        # Create sample CSV files
        sample_data = {
            "AAPL": [("2023-01-01", 150.0), ("2023-01-02", 151.0), ("2023-01-03", 152.0)],
            "GOOGL": [("2023-01-01", 2800.0), ("2023-01-02", 2810.0), ("2023-01-03", 2820.0)],
            "MSFT": [("2023-01-01", 300.0), ("2023-01-02", 301.0), ("2023-01-03", 302.0)],
        }
        
        for symbol, data in sample_data.items():
            csv_file = price_dir / f"{symbol}.csv"
            with open(csv_file, 'w') as f:
                f.write("date,close\n")
                for date, price in data:
                    f.write(f"{date},{price}\n")
        
        return sample_data
    
    def test_basket_manager_initialization(self, temp_dir):
        """Test that basket manager can be initialized."""
        manager = StockBasketManager(temp_dir)
        assert manager.data_dir == temp_dir
        assert manager.baskets_file == temp_dir / "stock_baskets.json"
        assert manager.price_dir == temp_dir / "price"
    
    def test_data_loader_with_basket_manager(self, temp_dir, sample_price_files):
        """Test that DataLoader integrates with basket manager."""
        data_loader = DataLoader(temp_dir / "price")
        
        # Check that basket manager is initialized
        assert hasattr(data_loader, 'basket_manager')
        assert isinstance(data_loader.basket_manager, StockBasketManager)
        
        # Check that available symbols are found
        symbols = data_loader.basket_manager.get_available_symbols()
        assert len(symbols) == 3
        assert set(symbols) == {"AAPL", "GOOGL", "MSFT"}
    
    def test_basket_creation_and_usage(self, temp_dir, sample_price_files):
        """Test creating a basket and using it with DataLoader."""
        data_loader = DataLoader(temp_dir / "price")
        
        # Create a basket
        basket_id = data_loader.basket_manager.create_basket(
            name="Test Tech Stocks",
            symbols=["AAPL", "GOOGL"],
            description="Test technology stocks",
            tags=["tech", "test"]
        )
        
        # Verify basket was created
        basket = data_loader.basket_manager.get_basket(basket_id)
        assert basket["name"] == "Test Tech Stocks"
        assert basket["symbols"] == ["AAPL", "GOOGL"]
        assert basket["description"] == "Test technology stocks"
        assert basket["tags"] == ["tech", "test"]
        
        # Load prices from basket
        prices = data_loader.load_prices_from_basket(basket_id)
        assert set(prices.columns) == {"AAPL", "GOOGL"}
        assert len(prices) == 3  # 3 days of data
    
    def test_basket_search_functionality(self, temp_dir, sample_price_files):
        """Test basket search functionality."""
        manager = StockBasketManager(temp_dir)
        
        # Create test baskets
        basket1_id = manager.create_basket("Tech Stocks", ["AAPL"], "Technology companies", ["tech"])
        basket2_id = manager.create_basket("Finance Stocks", ["GOOGL"], "Financial companies", ["finance"])
        basket3_id = manager.create_basket("Mixed Portfolio", ["AAPL", "GOOGL"], "Diversified portfolio", ["tech", "finance"])
        
        # Search by name
        results = manager.search_baskets(query="Tech")
        assert len(results) == 2  # "Tech Stocks" and "Mixed Portfolio" (has tech tag)
        
        # Search by tags
        results = manager.search_baskets(tags=["tech"])
        assert len(results) == 2  # "Tech Stocks" and "Mixed Portfolio"
        
        # Search by description
        results = manager.search_baskets(query="Financial")
        assert len(results) == 1  # "Finance Stocks"
    
    def test_basket_export_import(self, temp_dir, sample_price_files):
        """Test basket export and import functionality."""
        manager = StockBasketManager(temp_dir)
        
        # Create a basket
        basket_id = manager.create_basket(
            name="Export Test",
            symbols=["AAPL", "GOOGL"],
            description="Test export",
            tags=["test"]
        )
        
        # Export as JSON
        json_data = manager.export_basket(basket_id, "json")
        assert isinstance(json_data, dict)
        assert json_data["name"] == "Export Test"
        assert json_data["symbols"] == ["AAPL", "GOOGL"]
        
        # Export as CSV
        csv_data = manager.export_basket(basket_id, "csv")
        assert isinstance(csv_data, str)
        assert "symbol,basket_name,basket_id" in csv_data
        assert "AAPL,Export Test" in csv_data
        
        # Import from JSON (modify name to avoid duplicate)
        json_data["name"] = "Imported Export Test"
        new_basket_id = manager.import_basket(json_data, "json")
        new_basket = manager.get_basket(new_basket_id)
        assert new_basket["name"] == "Imported Export Test"
        assert new_basket["symbols"] == ["AAPL", "GOOGL"]
    
    def test_basket_statistics(self, temp_dir, sample_price_files):
        """Test basket statistics calculation."""
        manager = StockBasketManager(temp_dir)
        
        # Initially no baskets
        stats = manager.get_basket_stats()
        assert stats["total_baskets"] == 0
        assert stats["total_symbols"] == 0
        assert stats["unique_symbols"] == 0
        
        # Create some baskets
        manager.create_basket("Basket 1", ["AAPL", "GOOGL"])
        manager.create_basket("Basket 2", ["AAPL", "MSFT"])
        manager.create_basket("Basket 3", ["GOOGL", "MSFT"])
        
        # Get stats
        stats = manager.get_basket_stats()
        assert stats["total_baskets"] == 3
        assert stats["total_symbols"] == 6  # 2 + 2 + 2
        assert stats["unique_symbols"] == 3  # AAPL, GOOGL, MSFT
        assert stats["avg_symbols_per_basket"] == 2.0
        
        # Check most common symbols
        most_common = dict(stats["most_common_symbols"])
        assert most_common["AAPL"] == 2
        assert most_common["GOOGL"] == 2
        assert most_common["MSFT"] == 2
    
    def test_default_baskets_creation(self, temp_dir, sample_price_files):
        """Test creating default baskets."""
        manager = StockBasketManager(temp_dir)
        
        # Create default baskets
        created_baskets = manager.create_default_baskets()
        
        # Should create some baskets (those with valid symbols)
        assert len(created_baskets) > 0
        
        # Check that baskets were created
        all_baskets = manager.get_all_baskets()
        assert len(all_baskets) > 0
        
        # Verify at least one basket has the expected structure
        basket_names = [basket["name"] for basket in all_baskets.values()]
        assert any("Tech" in name for name in basket_names)
    
    def test_basket_validation(self, temp_dir, sample_price_files):
        """Test basket symbol validation."""
        manager = StockBasketManager(temp_dir)
        
        # Test with valid symbols
        validation = manager.validate_symbols(["AAPL", "GOOGL", "MSFT"])
        assert validation["valid"] == ["AAPL", "GOOGL", "MSFT"]
        assert validation["invalid"] == []
        
        # Test with mixed valid and invalid symbols
        validation = manager.validate_symbols(["AAPL", "INVALID_SYMBOL", "GOOGL"])
        assert validation["valid"] == ["AAPL", "GOOGL"]
        assert validation["invalid"] == ["INVALID_SYMBOL"]
        
        # Test with all invalid symbols
        validation = manager.validate_symbols(["INVALID1", "INVALID2"])
        assert validation["valid"] == []
        assert validation["invalid"] == ["INVALID1", "INVALID2"]
    
    def test_basket_crud_operations(self, temp_dir, sample_price_files):
        """Test complete CRUD operations for baskets."""
        manager = StockBasketManager(temp_dir)
        
        # Create
        basket_id = manager.create_basket(
            name="CRUD Test",
            symbols=["AAPL", "GOOGL"],
            description="Test CRUD operations",
            tags=["test", "crud"]
        )
        
        # Read
        basket = manager.get_basket(basket_id)
        assert basket["name"] == "CRUD Test"
        assert basket["symbols"] == ["AAPL", "GOOGL"]
        
        # Update
        success = manager.update_basket(
            basket_id,
            name="Updated CRUD Test",
            symbols=["AAPL", "GOOGL", "MSFT"],
            description="Updated description"
        )
        assert success is True
        
        # Verify update
        updated_basket = manager.get_basket(basket_id)
        assert updated_basket["name"] == "Updated CRUD Test"
        assert updated_basket["symbols"] == ["AAPL", "GOOGL", "MSFT"]
        assert updated_basket["description"] == "Updated description"
        
        # Delete
        success = manager.delete_basket(basket_id)
        assert success is True
        
        # Verify deletion
        deleted_basket = manager.get_basket(basket_id)
        assert deleted_basket is None
