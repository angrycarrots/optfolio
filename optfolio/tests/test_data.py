"""Tests for data loading and validation modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from ..data.loader import DataLoader
from ..data.validator import DataValidator


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "prices"
        self.data_dir.mkdir()
        
        # Create sample price data
        self.sample_data = self._create_sample_data()
        self._create_test_files()
        
    def teardown_method(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        data = {}
        
        for ticker in ['AAPL', 'GOOGL', 'MSFT']:
            # Generate realistic price data
            np.random.seed(42)  # For reproducibility
            prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
            data[ticker] = pd.DataFrame({
                'date': dates,
                'close': prices
            })
        
        return data
    
    def _create_test_files(self):
        """Create test CSV files."""
        for ticker, df in self.sample_data.items():
            file_path = self.data_dir / f"{ticker}.csv"
            df.to_csv(file_path, index=False)
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader(self.data_dir)
        assert loader.data_dir == self.data_dir
        assert loader._price_data is None
        assert loader._returns_data is None
    
    def test_load_prices_success(self):
        """Test successful price loading."""
        loader = DataLoader(self.data_dir)
        prices = loader.load_prices()
        
        assert isinstance(prices, pd.DataFrame)
        assert len(prices.columns) == 3  # AAPL, GOOGL, MSFT
        assert len(prices) > 0
        assert isinstance(prices.index, pd.DatetimeIndex)
        assert 'AAPL' in prices.columns
        assert 'GOOGL' in prices.columns
        assert 'MSFT' in prices.columns
    
    def test_load_prices_specific_tickers(self):
        """Test loading specific tickers."""
        loader = DataLoader(self.data_dir)
        prices = loader.load_prices(['AAPL', 'MSFT'])
        
        assert len(prices.columns) == 2
        assert 'AAPL' in prices.columns
        assert 'MSFT' in prices.columns
        assert 'GOOGL' not in prices.columns
    
    def test_load_prices_missing_ticker(self):
        """Test loading with missing ticker."""
        loader = DataLoader(self.data_dir)
        
        with pytest.raises(ValueError, match="Missing data for tickers"):
            loader.load_prices(['AAPL', 'INVALID'])
    
    def test_load_prices_no_data_directory(self):
        """Test loading with non-existent directory."""
        loader = DataLoader("non_existent_dir")
        
        with pytest.raises(FileNotFoundError):
            loader.load_prices()
    
    def test_get_returns_log(self):
        """Test log returns calculation."""
        loader = DataLoader(self.data_dir)
        loader.load_prices()
        returns = loader.get_returns(method="log")
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns.columns) == 3
        assert len(returns) > 0
        # First row should be NaN (no previous price)
        assert returns.iloc[0].isna().all()
    
    def test_get_returns_simple(self):
        """Test simple returns calculation."""
        loader = DataLoader(self.data_dir)
        loader.load_prices()
        returns = loader.get_returns(method="simple")
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns.columns) == 3
        assert len(returns) > 0
    
    def test_get_returns_invalid_method(self):
        """Test returns calculation with invalid method."""
        loader = DataLoader(self.data_dir)
        loader.load_prices()
        
        with pytest.raises(ValueError, match="Method must be"):
            loader.get_returns(method="invalid")
    
    def test_get_returns_no_price_data(self):
        """Test returns calculation without price data."""
        loader = DataLoader(self.data_dir)
        
        with pytest.raises(ValueError, match="Price data not loaded"):
            loader.get_returns()
    
    def test_get_available_tickers(self):
        """Test getting available tickers."""
        loader = DataLoader(self.data_dir)
        tickers = loader.get_available_tickers()
        
        assert isinstance(tickers, list)
        assert len(tickers) == 3
        assert 'AAPL' in tickers
        assert 'GOOGL' in tickers
        assert 'MSFT' in tickers
    
    def test_validate_data(self):
        """Test data validation."""
        loader = DataLoader(self.data_dir)
        loader.load_prices()
        issues = loader.validate_data()
        
        assert isinstance(issues, dict)
        assert 'warnings' in issues
        assert 'errors' in issues
    
    def test_get_data_info(self):
        """Test getting data information."""
        loader = DataLoader(self.data_dir)
        loader.load_prices()
        info = loader.get_data_info()
        
        assert isinstance(info, dict)
        assert 'tickers' in info
        assert 'date_range' in info
        assert 'total_observations' in info
        assert len(info['tickers']) == 3


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create clean price data
        self.clean_prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'GOOGL': 200 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'MSFT': 150 + np.cumsum(np.random.normal(0, 1, len(dates)))
        }, index=dates)
        
        # Create problematic price data
        self.problematic_prices = self.clean_prices.copy()
        # Add more missing data
        self.problematic_prices.loc['2021-01-01':'2021-01-10', 'AAPL'] = np.nan  # Missing data
        self.problematic_prices.loc['2021-01-02', 'GOOGL'] = 0  # Zero price
        self.problematic_prices.loc['2021-01-03', 'MSFT'] = -10  # Negative price
    
    def test_validate_price_data_clean(self):
        """Test validation of clean price data."""
        issues = DataValidator.validate_price_data(self.clean_prices)
        
        assert isinstance(issues, dict)
        assert 'errors' in issues
        assert 'warnings' in issues
        assert 'info' in issues
        assert len(issues['errors']) == 0  # No errors for clean data
    
    def test_validate_price_data_empty(self):
        """Test validation of empty price data."""
        empty_prices = pd.DataFrame()
        issues = DataValidator.validate_price_data(empty_prices)
        
        assert len(issues['errors']) > 0
        assert "Price data is empty" in issues['errors']
    
    def test_validate_price_data_missing_values(self):
        """Test validation with missing values."""
        issues = DataValidator.validate_price_data(self.problematic_prices)
        
        assert len(issues['warnings']) > 0
        # Should have warnings about missing data, zero prices, etc.
    
    def test_validate_price_data_wrong_index(self):
        """Test validation with wrong index type."""
        wrong_index_prices = self.clean_prices.copy()
        wrong_index_prices.index = range(len(wrong_index_prices))
        
        issues = DataValidator.validate_price_data(wrong_index_prices)
        
        assert len(issues['errors']) > 0
        assert "Index must be DatetimeIndex" in issues['errors']
    
    def test_validate_returns_data(self):
        """Test returns data validation."""
        returns = self.clean_prices.pct_change()
        issues = DataValidator.validate_returns_data(returns)
        
        assert isinstance(issues, dict)
        assert 'errors' in issues
        assert 'warnings' in issues
        assert 'info' in issues
    
    def test_validate_returns_data_extreme_values(self):
        """Test returns validation with extreme values."""
        extreme_returns = self.clean_prices.pct_change()
        extreme_returns.loc['2021-01-01', 'AAPL'] = 0.6  # 60% return
        
        issues = DataValidator.validate_returns_data(extreme_returns)
        
        assert len(issues['warnings']) > 0
        # Should have warnings about extreme returns
    
    def test_check_data_consistency(self):
        """Test data consistency check."""
        prices = self.clean_prices
        returns = prices.pct_change()
        
        issues = DataValidator.check_data_consistency(prices, returns)
        
        assert isinstance(issues, dict)
        assert 'errors' in issues
        assert 'warnings' in issues
        assert 'info' in issues
    
    def test_get_data_quality_score(self):
        """Test data quality score calculation."""
        score = DataValidator.get_data_quality_score(self.clean_prices)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0.8  # Clean data should have high score
    
    def test_get_data_quality_score_problematic(self):
        """Test data quality score for problematic data."""
        score = DataValidator.get_data_quality_score(self.problematic_prices)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score < 0.999  # Problematic data should have lower score
    
    def test_suggest_data_cleaning(self):
        """Test data cleaning suggestions."""
        suggestions = DataValidator.suggest_data_cleaning(self.problematic_prices)
        
        assert isinstance(suggestions, dict)
        assert 'required' in suggestions
        assert 'recommended' in suggestions
        assert 'optional' in suggestions
        
        # Should have suggestions for problematic data
        assert len(suggestions['required']) > 0 or len(suggestions['recommended']) > 0


class TestDataLoaderIntegration:
    """Integration tests for data loading."""
    
    def setup_method(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "prices"
        self.data_dir.mkdir()
        
        # Create more realistic test data
        self._create_realistic_test_data()
    
    def teardown_method(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_realistic_test_data(self):
        """Create realistic test data with various scenarios."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Create different scenarios
        scenarios = {
            'AAPL': {'start': 100, 'volatility': 0.02, 'trend': 0.0001},
            'GOOGL': {'start': 200, 'volatility': 0.025, 'trend': 0.0002},
            'MSFT': {'start': 150, 'volatility': 0.018, 'trend': 0.00015},
            'TSLA': {'start': 50, 'volatility': 0.04, 'trend': 0.0003}
        }
        
        for ticker, params in scenarios.items():
            np.random.seed(hash(ticker) % 1000)  # Different seed for each ticker
            
            # Generate price series with trend and volatility
            returns = np.random.normal(params['trend'], params['volatility'], len(dates))
            prices = params['start'] * np.exp(np.cumsum(returns))
            
            # Add some missing data
            missing_indices = np.random.choice(len(dates), size=len(dates)//20, replace=False)
            prices[missing_indices] = np.nan
            
            df = pd.DataFrame({
                'date': dates,
                'close': prices
            })
            
            file_path = self.data_dir / f"{ticker}.csv"
            df.to_csv(file_path, index=False)
    
    def test_full_workflow(self):
        """Test complete data loading workflow."""
        loader = DataLoader(self.data_dir)
        
        # Load prices
        prices = loader.load_prices()
        assert len(prices.columns) == 4
        assert len(prices) > 0
        
        # Get returns
        returns = loader.get_returns()
        assert len(returns.columns) == 4
        assert len(returns) > 0
        
        # Validate data
        issues = loader.validate_data()
        assert isinstance(issues, dict)
        
        # Get data info
        info = loader.get_data_info()
        assert isinstance(info, dict)
        assert info['total_observations'] > 0
    
    def test_data_validation_workflow(self):
        """Test data validation workflow."""
        loader = DataLoader(self.data_dir)
        prices = loader.load_prices()
        returns = loader.get_returns()
        
        # Validate price data
        price_issues = DataValidator.validate_price_data(prices)
        assert isinstance(price_issues, dict)
        
        # Validate returns data
        returns_issues = DataValidator.validate_returns_data(returns)
        assert isinstance(returns_issues, dict)
        
        # Check consistency
        consistency_issues = DataValidator.check_data_consistency(prices, returns)
        assert isinstance(consistency_issues, dict)
        
        # Get quality score
        quality_score = DataValidator.get_data_quality_score(prices)
        assert 0 <= quality_score <= 1
        
        # Get cleaning suggestions
        suggestions = DataValidator.suggest_data_cleaning(prices)
        assert isinstance(suggestions, dict)
