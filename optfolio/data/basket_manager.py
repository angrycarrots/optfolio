"""Stock basket management module for portfolio analysis."""

import json
import uuid
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd


class StockBasketManager:
    """Manage stock baskets (collections of stocks/ETFs) with JSON storage."""
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initialize the basket manager.
        
        Args:
            data_dir: Directory containing basket data and price files
        """
        self.data_dir = Path(data_dir)
        self.baskets_file = self.data_dir / "stock_baskets.json"
        self.price_dir = self.data_dir / "price"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize baskets file if it doesn't exist
        if not self.baskets_file.exists():
            self._initialize_baskets_file()
    
    def _initialize_baskets_file(self):
        """Initialize the baskets JSON file with default structure."""
        default_baskets = {
            "baskets": {},
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat()
            }
        }
        self._save_baskets(default_baskets)
    
    def _load_baskets(self) -> Dict[str, Any]:
        """Load baskets from JSON file."""
        try:
            with open(self.baskets_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading baskets: {e}")
            return {"baskets": {}, "metadata": {"version": "1.0"}}
    
    def _save_baskets(self, baskets_data: Dict[str, Any]):
        """Save baskets to JSON file."""
        baskets_data["metadata"]["last_modified"] = datetime.now().isoformat()
        with open(self.baskets_file, 'w') as f:
            json.dump(baskets_data, f, indent=2)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from price data directory."""
        if not self.price_dir.exists():
            return []
        
        csv_files = list(self.price_dir.glob("*.csv"))
        return [f.stem for f in csv_files]
    
    def validate_symbols(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Validate symbols against available data.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary with 'valid' and 'invalid' symbol lists
        """
        available_symbols = set(self.get_available_symbols())
        
        # Preserve original order for valid symbols
        valid_symbols = [symbol for symbol in symbols if symbol in available_symbols]
        invalid_symbols = [symbol for symbol in symbols if symbol not in available_symbols]
        
        return {
            "valid": valid_symbols,
            "invalid": invalid_symbols
        }
    
    def create_basket(self, name: str, symbols: List[str], 
                     description: Optional[str] = None, 
                     tags: Optional[List[str]] = None) -> str:
        """Create a new stock basket.
        
        Args:
            name: Basket display name
            symbols: List of stock/ETF symbols
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            Basket ID
            
        Raises:
            ValueError: If basket name already exists or no valid symbols
        """
        # Validate symbols
        validation = self.validate_symbols(symbols)
        if not validation["valid"]:
            raise ValueError(f"No valid symbols found. Invalid symbols: {validation['invalid']}")
        
        # Check for duplicate names
        baskets_data = self._load_baskets()
        existing_names = {basket["name"] for basket in baskets_data["baskets"].values()}
        if name in existing_names:
            raise ValueError(f"Basket with name '{name}' already exists")
        
        # Generate unique ID
        basket_id = str(uuid.uuid4())
        
        # Create basket
        basket = {
            "id": basket_id,
            "name": name,
            "description": description or "",
            "symbols": validation["valid"],  # Only include valid symbols
            "created_date": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "tags": tags or []
        }
        
        # Save to file
        baskets_data["baskets"][basket_id] = basket
        self._save_baskets(baskets_data)
        
        return basket_id
    
    def get_basket(self, basket_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific basket by ID.
        
        Args:
            basket_id: Basket identifier
            
        Returns:
            Basket data or None if not found
        """
        baskets_data = self._load_baskets()
        return baskets_data["baskets"].get(basket_id)
    
    def get_all_baskets(self) -> Dict[str, Dict[str, Any]]:
        """Get all baskets.
        
        Returns:
            Dictionary of all baskets keyed by basket ID
        """
        baskets_data = self._load_baskets()
        return baskets_data["baskets"]
    
    def update_basket(self, basket_id: str, **updates) -> bool:
        """Update an existing basket.
        
        Args:
            basket_id: Basket identifier
            **updates: Fields to update (name, symbols, description, tags)
            
        Returns:
            True if updated successfully, False if basket not found
            
        Raises:
            ValueError: If validation fails
        """
        baskets_data = self._load_baskets()
        if basket_id not in baskets_data["baskets"]:
            return False
        
        basket = baskets_data["baskets"][basket_id]
        
        # Validate symbols if provided
        if "symbols" in updates:
            validation = self.validate_symbols(updates["symbols"])
            if not validation["valid"]:
                raise ValueError(f"No valid symbols found. Invalid symbols: {validation['invalid']}")
            updates["symbols"] = validation["valid"]
        
        # Check for duplicate names if name is being updated
        if "name" in updates:
            existing_names = {b["name"] for bid, b in baskets_data["baskets"].items() if bid != basket_id}
            if updates["name"] in existing_names:
                raise ValueError(f"Basket with name '{updates['name']}' already exists")
        
        # Update basket
        basket.update(updates)
        basket["last_modified"] = datetime.now().isoformat()
        
        self._save_baskets(baskets_data)
        return True
    
    def delete_basket(self, basket_id: str) -> bool:
        """Delete a basket.
        
        Args:
            basket_id: Basket identifier
            
        Returns:
            True if deleted successfully, False if basket not found
        """
        baskets_data = self._load_baskets()
        if basket_id not in baskets_data["baskets"]:
            return False
        
        del baskets_data["baskets"][basket_id]
        self._save_baskets(baskets_data)
        return True
    
    def search_baskets(self, query: str = "", tags: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Search baskets by name, description, or tags.
        
        Args:
            query: Text query to search in name and description
            tags: List of tags to filter by
            
        Returns:
            Dictionary of matching baskets
        """
        all_baskets = self.get_all_baskets()
        query_lower = query.lower()
        
        filtered_baskets = {}
        for basket_id, basket in all_baskets.items():
            # Text search
            if query:
                name_match = query_lower in basket["name"].lower()
                desc_match = query_lower in basket.get("description", "").lower()
                # Also search in tags
                tag_match = any(query_lower in tag.lower() for tag in basket.get("tags", []))
                if not (name_match or desc_match or tag_match):
                    continue
            
            # Tag filter
            if tags:
                basket_tags = set(basket.get("tags", []))
                if not any(tag in basket_tags for tag in tags):
                    continue
            
            filtered_baskets[basket_id] = basket
        
        return filtered_baskets
    
    def get_basket_stats(self) -> Dict[str, Any]:
        """Get statistics about all baskets.
        
        Returns:
            Dictionary with basket statistics
        """
        baskets = self.get_all_baskets()
        
        if not baskets:
            return {
                "total_baskets": 0,
                "total_symbols": 0,
                "unique_symbols": 0,
                "avg_symbols_per_basket": 0,
                "most_common_symbols": []
            }
        
        all_symbols = []
        symbol_counts = {}
        available_symbols = set(self.get_available_symbols())
        
        for basket in baskets.values():
            symbols = basket.get("symbols", [])
            # Only count symbols that are actually available in price data
            valid_symbols = [symbol for symbol in symbols if symbol in available_symbols]
            all_symbols.extend(valid_symbols)
            for symbol in valid_symbols:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Get most common symbols
        most_common = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_baskets": len(baskets),
            "total_symbols": len(all_symbols),
            "unique_symbols": len(set(all_symbols)),
            "avg_symbols_per_basket": len(all_symbols) / len(baskets) if baskets else 0,
            "most_common_symbols": most_common
        }
    
    def export_basket(self, basket_id: str, format: str = "json") -> Union[Dict[str, Any], str]:
        """Export a basket in specified format.
        
        Args:
            basket_id: Basket identifier
            format: Export format ("json" or "csv")
            
        Returns:
            Exported basket data
            
        Raises:
            ValueError: If basket not found or invalid format
        """
        basket = self.get_basket(basket_id)
        if not basket:
            raise ValueError(f"Basket {basket_id} not found")
        
        if format == "json":
            return basket
        elif format == "csv":
            # Create CSV format with symbols
            df = pd.DataFrame({
                "symbol": basket["symbols"],
                "basket_name": [basket["name"]] * len(basket["symbols"]),
                "basket_id": [basket_id] * len(basket["symbols"])
            })
            return df.to_csv(index=False)
        else:
            raise ValueError("Format must be 'json' or 'csv'")
    
    def import_basket(self, basket_data: Union[Dict[str, Any], str], 
                     format: str = "json") -> str:
        """Import a basket from external data.
        
        Args:
            basket_data: Basket data (dict for JSON, string for CSV)
            format: Import format ("json" or "csv")
            
        Returns:
            Created basket ID
            
        Raises:
            ValueError: If import data is invalid
        """
        if format == "json":
            if isinstance(basket_data, str):
                basket_data = json.loads(basket_data)
            
            return self.create_basket(
                name=basket_data["name"],
                symbols=basket_data["symbols"],
                description=basket_data.get("description"),
                tags=basket_data.get("tags")
            )
        
        elif format == "csv":
            if isinstance(basket_data, str):
                df = pd.read_csv(io.StringIO(basket_data))
            else:
                df = basket_data
            
            # Group by basket name
            basket_groups = df.groupby("basket_name")
            
            basket_ids = []
            for basket_name, group in basket_groups:
                symbols = group["symbol"].tolist()
                basket_id = self.create_basket(
                    name=basket_name,
                    symbols=symbols
                )
                basket_ids.append(basket_id)
            
            return basket_ids[0] if len(basket_ids) == 1 else basket_ids
        
        else:
            raise ValueError("Format must be 'json' or 'csv'")
    
    def create_default_baskets(self):
        """Create default baskets for common use cases."""
        default_baskets = [
            {
                "name": "S&P 500 Tech",
                "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
                "description": "Major technology stocks from S&P 500",
                "tags": ["tech", "large-cap", "sp500"]
            },
            {
                "name": "Sector ETFs",
                "symbols": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "IYR"],
                "description": "Sector SPDR ETFs for broad market exposure",
                "tags": ["etf", "sectors", "diversified"]
            },
            {
                "name": "Financial Services",
                "symbols": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "XLF", "KBE", "KRE"],
                "description": "Major financial services companies and ETFs",
                "tags": ["financials", "banks", "etf"]
            },
            {
                "name": "Healthcare & Biotech",
                "symbols": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "XLV", "IBB"],
                "description": "Healthcare and biotechnology companies",
                "tags": ["healthcare", "biotech", "pharma"]
            }
        ]
        
        created_baskets = []
        for basket_data in default_baskets:
            try:
                # Check if basket already exists
                existing_baskets = self.get_all_baskets()
                existing_names = {b["name"] for b in existing_baskets.values()}
                
                if basket_data["name"] not in existing_names:
                    basket_id = self.create_basket(**basket_data)
                    created_baskets.append(basket_id)
            except ValueError as e:
                print(f"Could not create default basket '{basket_data['name']}': {e}")
        
        return created_baskets
