"""Tests for the StockBasketManager class."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from optfolio.data.basket_manager import StockBasketManager


class TestStockBasketManager:
    """Test cases for StockBasketManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def basket_manager(self, temp_dir):
        """Create a StockBasketManager instance for testing."""
        return StockBasketManager(temp_dir)
    
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
            "INVALID": [("2023-01-01", 100.0), ("2023-01-02", 101.0)]  # Will be used for testing invalid symbols
        }
        
        for symbol, data in sample_data.items():
            csv_file = price_dir / f"{symbol}.csv"
            with open(csv_file, 'w') as f:
                f.write("date,close\n")
                for date, price in data:
                    f.write(f"{date},{price}\n")
        
        return sample_data
    
    def test_initialization(self, temp_dir):
        """Test StockBasketManager initialization."""
        manager = StockBasketManager(temp_dir)
        
        assert manager.data_dir == temp_dir
        assert manager.baskets_file == temp_dir / "stock_baskets.json"
        assert manager.price_dir == temp_dir / "price"
        
        # Check that baskets file is created
        assert manager.baskets_file.exists()
        
        # Check initial structure
        with open(manager.baskets_file, 'r') as f:
            data = json.load(f)
            assert "baskets" in data
            assert "metadata" in data
            assert data["baskets"] == {}
    
    def test_get_available_symbols(self, basket_manager, sample_price_files):
        """Test getting available symbols from price directory."""
        symbols = basket_manager.get_available_symbols()
        
        expected_symbols = ["AAPL", "GOOGL", "MSFT", "INVALID"]
        assert set(symbols) == set(expected_symbols)
    
    def test_validate_symbols(self, basket_manager, sample_price_files):
        """Test symbol validation."""
        # Test with valid symbols
        valid_symbols = ["AAPL", "GOOGL", "MSFT"]
        result = basket_manager.validate_symbols(valid_symbols)
        
        assert result["valid"] == valid_symbols
        assert result["invalid"] == []
        
        # Test with mixed valid and invalid symbols
        mixed_symbols = ["AAPL", "INVALID_SYMBOL", "GOOGL", "MISSING"]
        result = basket_manager.validate_symbols(mixed_symbols)
        
        assert set(result["valid"]) == {"AAPL", "GOOGL"}
        assert set(result["invalid"]) == {"INVALID_SYMBOL", "MISSING"}
    
    def test_create_basket(self, basket_manager, sample_price_files):
        """Test creating a new basket."""
        basket_id = basket_manager.create_basket(
            name="Tech Stocks",
            symbols=["AAPL", "GOOGL", "MSFT"],
            description="Major technology companies",
            tags=["tech", "large-cap"]
        )
        
        # Check that basket ID is returned
        assert basket_id is not None
        assert len(basket_id) > 0
        
        # Check that basket is saved
        basket = basket_manager.get_basket(basket_id)
        assert basket is not None
        assert basket["name"] == "Tech Stocks"
        assert basket["symbols"] == ["AAPL", "GOOGL", "MSFT"]
        assert basket["description"] == "Major technology companies"
        assert basket["tags"] == ["tech", "large-cap"]
        assert "created_date" in basket
        assert "last_modified" in basket
    
    def test_create_basket_duplicate_name(self, basket_manager, sample_price_files):
        """Test creating basket with duplicate name."""
        # Create first basket
        basket_manager.create_basket("Tech Stocks", ["AAPL", "GOOGL"])
        
        # Try to create second basket with same name
        with pytest.raises(ValueError, match="Basket with name 'Tech Stocks' already exists"):
            basket_manager.create_basket("Tech Stocks", ["MSFT", "TSLA"])
    
    def test_create_basket_invalid_symbols(self, basket_manager, sample_price_files):
        """Test creating basket with invalid symbols."""
        with pytest.raises(ValueError, match="No valid symbols found"):
            basket_manager.create_basket("Invalid Basket", ["INVALID_SYMBOL", "MISSING"])
    
    def test_get_basket(self, basket_manager, sample_price_files):
        """Test getting a specific basket."""
        # Create a basket
        basket_id = basket_manager.create_basket("Test Basket", ["AAPL", "GOOGL"])
        
        # Get the basket
        basket = basket_manager.get_basket(basket_id)
        assert basket is not None
        assert basket["name"] == "Test Basket"
        assert basket["symbols"] == ["AAPL", "GOOGL"]
        
        # Test getting non-existent basket
        non_existent = basket_manager.get_basket("non-existent-id")
        assert non_existent is None
    
    def test_get_all_baskets(self, basket_manager, sample_price_files):
        """Test getting all baskets."""
        # Initially no baskets
        baskets = basket_manager.get_all_baskets()
        assert len(baskets) == 0
        
        # Create some baskets
        basket1_id = basket_manager.create_basket("Basket 1", ["AAPL"])
        basket2_id = basket_manager.create_basket("Basket 2", ["GOOGL"])
        
        # Get all baskets
        baskets = basket_manager.get_all_baskets()
        assert len(baskets) == 2
        assert basket1_id in baskets
        assert basket2_id in baskets
        assert baskets[basket1_id]["name"] == "Basket 1"
        assert baskets[basket2_id]["name"] == "Basket 2"
    
    def test_update_basket(self, basket_manager, sample_price_files):
        """Test updating a basket."""
        # Create a basket
        basket_id = basket_manager.create_basket("Original Name", ["AAPL"], "Original description")
        
        # Update the basket
        success = basket_manager.update_basket(
            basket_id,
            name="Updated Name",
            symbols=["AAPL", "GOOGL"],
            description="Updated description",
            tags=["tech"]
        )
        
        assert success is True
        
        # Check the updated basket
        basket = basket_manager.get_basket(basket_id)
        assert basket["name"] == "Updated Name"
        assert basket["symbols"] == ["AAPL", "GOOGL"]
        assert basket["description"] == "Updated description"
        assert basket["tags"] == ["tech"]
        assert basket["last_modified"] != basket["created_date"]
    
    def test_update_basket_invalid_symbols(self, basket_manager, sample_price_files):
        """Test updating basket with invalid symbols."""
        basket_id = basket_manager.create_basket("Test Basket", ["AAPL"])
        
        with pytest.raises(ValueError, match="No valid symbols found"):
            basket_manager.update_basket(basket_id, symbols=["INVALID_SYMBOL"])
    
    def test_update_basket_duplicate_name(self, basket_manager, sample_price_files):
        """Test updating basket with duplicate name."""
        basket1_id = basket_manager.create_basket("Basket 1", ["AAPL"])
        basket2_id = basket_manager.create_basket("Basket 2", ["GOOGL"])
        
        with pytest.raises(ValueError, match="Basket with name 'Basket 1' already exists"):
            basket_manager.update_basket(basket2_id, name="Basket 1")
    
    def test_update_basket_not_found(self, basket_manager):
        """Test updating non-existent basket."""
        success = basket_manager.update_basket("non-existent-id", name="New Name")
        assert success is False
    
    def test_delete_basket(self, basket_manager, sample_price_files):
        """Test deleting a basket."""
        # Create a basket
        basket_id = basket_manager.create_basket("To Delete", ["AAPL"])
        
        # Verify it exists
        assert basket_manager.get_basket(basket_id) is not None
        
        # Delete the basket
        success = basket_manager.delete_basket(basket_id)
        assert success is True
        
        # Verify it's deleted
        assert basket_manager.get_basket(basket_id) is None
        
        # Try to delete non-existent basket
        success = basket_manager.delete_basket("non-existent-id")
        assert success is False
    
    def test_search_baskets(self, basket_manager, sample_price_files):
        """Test searching baskets."""
        # Create test baskets
        basket1_id = basket_manager.create_basket("Tech Stocks", ["AAPL"], "Technology companies", ["tech"])
        basket2_id = basket_manager.create_basket("Finance Stocks", ["GOOGL"], "Financial companies", ["finance"])
        basket3_id = basket_manager.create_basket("Mixed Portfolio", ["AAPL", "GOOGL"], "Diversified portfolio", ["tech", "finance"])
        
        # Search by name
        results = basket_manager.search_baskets(query="Tech")
        assert len(results) == 2  # "Tech Stocks" and "Mixed Portfolio"
        assert basket1_id in results
        assert basket3_id in results
        
        # Search by description
        results = basket_manager.search_baskets(query="Financial")
        assert len(results) == 1
        assert basket2_id in results
        
        # Search by tags
        results = basket_manager.search_baskets(tags=["tech"])
        assert len(results) == 2  # "Tech Stocks" and "Mixed Portfolio"
        
        # Search with both query and tags
        results = basket_manager.search_baskets(query="Portfolio", tags=["tech"])
        assert len(results) == 1
        assert basket3_id in results
    
    def test_get_basket_stats(self, basket_manager, sample_price_files):
        """Test getting basket statistics."""
        # Initially no baskets
        stats = basket_manager.get_basket_stats()
        assert stats["total_baskets"] == 0
        assert stats["total_symbols"] == 0
        assert stats["unique_symbols"] == 0
        assert stats["avg_symbols_per_basket"] == 0
        assert stats["most_common_symbols"] == []
        
        # Create some baskets
        basket_manager.create_basket("Basket 1", ["AAPL", "GOOGL"])
        basket_manager.create_basket("Basket 2", ["AAPL", "MSFT"])
        basket_manager.create_basket("Basket 3", ["GOOGL", "MSFT", "INVALID"])
        
        # Get stats
        stats = basket_manager.get_basket_stats()
        assert stats["total_baskets"] == 3
        assert stats["total_symbols"] == 7  # 2 + 2 + 3
        assert stats["unique_symbols"] == 4  # AAPL, GOOGL, MSFT, INVALID (all are in test price files)
        assert stats["avg_symbols_per_basket"] == 7 / 3
        
        # Check most common symbols
        most_common = dict(stats["most_common_symbols"])
        assert most_common["AAPL"] == 2
        assert most_common["GOOGL"] == 2
        assert most_common["MSFT"] == 2
    
    def test_export_basket_json(self, basket_manager, sample_price_files):
        """Test exporting basket as JSON."""
        basket_id = basket_manager.create_basket("Export Test", ["AAPL", "GOOGL"], "Test basket", ["test"])
        
        exported = basket_manager.export_basket(basket_id, "json")
        
        assert isinstance(exported, dict)
        assert exported["name"] == "Export Test"
        assert exported["symbols"] == ["AAPL", "GOOGL"]
        assert exported["description"] == "Test basket"
        assert exported["tags"] == ["test"]
    
    def test_export_basket_csv(self, basket_manager, sample_price_files):
        """Test exporting basket as CSV."""
        basket_id = basket_manager.create_basket("Export Test", ["AAPL", "GOOGL"])
        
        exported = basket_manager.export_basket(basket_id, "csv")
        
        assert isinstance(exported, str)
        lines = exported.strip().split('\n')
        assert len(lines) == 3  # Header + 2 symbols
        assert "symbol,basket_name,basket_id" in lines[0]
        assert "AAPL,Export Test" in lines[1]
        assert "GOOGL,Export Test" in lines[2]
    
    def test_export_basket_invalid_format(self, basket_manager, sample_price_files):
        """Test exporting basket with invalid format."""
        basket_id = basket_manager.create_basket("Test", ["AAPL"])
        
        with pytest.raises(ValueError, match="Format must be 'json' or 'csv'"):
            basket_manager.export_basket(basket_id, "xml")
    
    def test_export_basket_not_found(self, basket_manager):
        """Test exporting non-existent basket."""
        with pytest.raises(ValueError, match="Basket non-existent-id not found"):
            basket_manager.export_basket("non-existent-id", "json")
    
    def test_import_basket_json(self, basket_manager, sample_price_files):
        """Test importing basket from JSON."""
        basket_data = {
            "name": "Imported Basket",
            "symbols": ["AAPL", "GOOGL"],
            "description": "Imported from JSON",
            "tags": ["imported"]
        }
        
        basket_id = basket_manager.import_basket(basket_data, "json")
        
        # Verify the basket was created
        basket = basket_manager.get_basket(basket_id)
        assert basket["name"] == "Imported Basket"
        assert basket["symbols"] == ["AAPL", "GOOGL"]
        assert basket["description"] == "Imported from JSON"
        assert basket["tags"] == ["imported"]
    
    def test_import_basket_csv(self, basket_manager, sample_price_files):
        """Test importing basket from CSV."""
        csv_data = "symbol,basket_name\nAAPL,Imported Basket\nGOOGL,Imported Basket"
        
        basket_id = basket_manager.import_basket(csv_data, "csv")
        
        # Verify the basket was created
        basket = basket_manager.get_basket(basket_id)
        assert basket["name"] == "Imported Basket"
        assert set(basket["symbols"]) == {"AAPL", "GOOGL"}
    
    def test_import_basket_invalid_format(self, basket_manager):
        """Test importing basket with invalid format."""
        with pytest.raises(ValueError, match="Format must be 'json' or 'csv'"):
            basket_manager.import_basket("data", "xml")
    
    def test_create_default_baskets(self, basket_manager, sample_price_files):
        """Test creating default baskets."""
        # Create some of the symbols that default baskets would use
        # (We'll create a few that exist in our sample data)
        
        created_baskets = basket_manager.create_default_baskets()
        
        # Should create some baskets (those with valid symbols)
        assert len(created_baskets) > 0
        
        # Check that baskets were created
        all_baskets = basket_manager.get_all_baskets()
        assert len(all_baskets) > 0
        
        # Verify at least one basket has the expected structure
        basket_names = [basket["name"] for basket in all_baskets.values()]
        assert any("Tech" in name for name in basket_names)
    
    def test_basket_persistence(self, temp_dir, sample_price_files):
        """Test that baskets persist across manager instances."""
        # Create first manager and basket
        manager1 = StockBasketManager(temp_dir)
        basket_id = manager1.create_basket("Persistent Basket", ["AAPL", "GOOGL"])
        
        # Create second manager (simulating restart)
        manager2 = StockBasketManager(temp_dir)
        
        # Verify basket still exists
        basket = manager2.get_basket(basket_id)
        assert basket is not None
        assert basket["name"] == "Persistent Basket"
        assert basket["symbols"] == ["AAPL", "GOOGL"]
