"""Data loading and preprocessing module for portfolio analysis."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .basket_manager import StockBasketManager


class DataLoader:
    """Load and preprocess financial data from CSV files."""
    
    def __init__(self, data_dir: Union[str, Path] = "prices"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing CSV price data files
        """
        self.data_dir = Path(data_dir)
        self._price_data: Optional[pd.DataFrame] = None
        self._returns_data: Optional[pd.DataFrame] = None
        
        # Initialize basket manager if data directory contains basket data
        basket_data_dir = self.data_dir.parent if self.data_dir.name == "price" else self.data_dir
        self.basket_manager = StockBasketManager(basket_data_dir)
        
    def load_prices(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Load price data for specified tickers.
        
        Args:
            tickers: List of ticker symbols to load. If None, loads all available.
            
        Returns:
            DataFrame with date index and ticker columns containing close prices
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If no valid CSV files found
        """
        # if not self.data_dir.exists():
        #     raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
            
        # Get list of CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
            
        # Filter by tickers if specified
        if tickers:
            available_tickers = {f.stem for f in csv_files}
            missing_tickers = set(tickers) - available_tickers
            if missing_tickers:
                raise ValueError(f"Missing data for tickers: {missing_tickers}")
            csv_files = [f for f in csv_files if f.stem in tickers]
        
        # Load each CSV file
        price_data = {}
        for csv_file in csv_files:
            ticker = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                if 'date' not in df.columns or 'close' not in df.columns:
                    print(f"Warning: {csv_file} missing required columns (date, close)")
                    continue
                    
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')['close'].rename(ticker)
                price_data[ticker] = df
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        if not price_data:
            raise ValueError("No valid price data could be loaded")
            
        # Combine all tickers into a single DataFrame
        self._price_data = pd.DataFrame(price_data)
        self._price_data = self._price_data.sort_index()
        
        return self._price_data
    
    def get_returns(self, method: str = "log") -> pd.DataFrame:
        """Calculate returns from price data.
        
        Args:
            method: Return calculation method ("log" or "simple")
            
        Returns:
            DataFrame with returns data
            
        Raises:
            ValueError: If price data not loaded or invalid method
        """
        if self._price_data is None:
            raise ValueError("Price data not loaded. Call load_prices() first.")
            
        if method not in ["log", "simple"]:
            raise ValueError("Method must be 'log' or 'simple'")
            
        if method == "log":
            self._returns_data = np.log(self._price_data / self._price_data.shift(1))
        else:
            self._returns_data = self._price_data.pct_change()
            
        return self._returns_data
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available ticker symbols.
        
        Returns:
            List of ticker symbols with available data
        """
        if not self.data_dir.exists():
            return []
            
        csv_files = list(self.data_dir.glob("*.csv"))
        return [f.stem for f in csv_files]
    
    def validate_data(self) -> Dict[str, List[str]]:
        """Validate loaded data for common issues.
        
        Returns:
            Dictionary with validation results
        """
        if self._price_data is None:
            return {"errors": ["No data loaded"]}
            
        issues = {
            "warnings": [],
            "errors": []
        }
        
        # Check for missing values
        missing_pct = self._price_data.isnull().sum() / len(self._price_data) * 100
        high_missing = missing_pct[missing_pct > 10]
        if not high_missing.empty:
            issues["warnings"].append(f"High missing data: {high_missing.to_dict()}")
            
        # Check for zero or negative prices
        zero_prices = (self._price_data <= 0).sum()
        if zero_prices.any():
            issues["warnings"].append(f"Zero/negative prices found: {zero_prices[zero_prices > 0].to_dict()}")
            
        # Check for data gaps
        for ticker in self._price_data.columns:
            price_series = self._price_data[ticker].dropna()
            if len(price_series) > 1:
                gaps = price_series.index.to_series().diff().dt.days > 7
                if gaps.any():
                    issues["warnings"].append(f"Data gaps detected for {ticker}")
                    
        return issues
    
    def get_data_info(self) -> Dict:
        """Get information about loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if self._price_data is None:
            return {"error": "No data loaded"}
            
        info = {
            "tickers": list(self._price_data.columns),
            "date_range": {
                "start": self._price_data.index.min().strftime("%Y-%m-%d"),
                "end": self._price_data.index.max().strftime("%Y-%m-%d")
            },
            "total_observations": len(self._price_data),
            "missing_data": self._price_data.isnull().sum().to_dict(),
            "data_completeness": (1 - self._price_data.isnull().sum() / len(self._price_data)).to_dict()
        }
        
        return info
    
    def load_prices_from_basket(self, basket_id: str) -> pd.DataFrame:
        """Load price data for symbols in a specific basket.
        
        Args:
            basket_id: Basket identifier
            
        Returns:
            DataFrame with price data for basket symbols
            
        Raises:
            ValueError: If basket not found or no valid symbols
        """
        basket = self.basket_manager.get_basket(basket_id)
        if not basket:
            raise ValueError(f"Basket {basket_id} not found")
        
        symbols = basket.get("symbols", [])
        if not symbols:
            raise ValueError(f"Basket {basket_id} contains no symbols")
        
        return self.load_prices(tickers=symbols)
    
    def get_basket_info(self, basket_id: str) -> Optional[Dict]:
        """Get information about a specific basket.
        
        Args:
            basket_id: Basket identifier
            
        Returns:
            Basket information or None if not found
        """
        return self.basket_manager.get_basket(basket_id)
    
    def get_all_baskets(self) -> Dict[str, Dict]:
        """Get all available baskets.
        
        Returns:
            Dictionary of all baskets
        """
        return self.basket_manager.get_all_baskets()
    
    def get_basket_stats(self) -> Dict:
        """Get statistics about all baskets.
        
        Returns:
            Dictionary with basket statistics
        """
        return self.basket_manager.get_basket_stats()
