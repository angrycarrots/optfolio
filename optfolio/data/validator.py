"""Data validation module for portfolio analysis."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataValidator:
    """Validate financial data for portfolio analysis."""
    
    @staticmethod
    def validate_price_data(prices: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate price data for common issues.
        
        Args:
            prices: DataFrame with date index and ticker columns
            
        Returns:
            Dictionary with validation results
        """
        issues = {
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        if prices.empty:
            issues["errors"].append("Price data is empty")
            return issues
            
        # Check for required columns
        if len(prices.columns) == 0:
            issues["errors"].append("No ticker columns found")
            
        # Check for required index
        if not isinstance(prices.index, pd.DatetimeIndex):
            issues["errors"].append("Index must be DatetimeIndex")
            
        # Check for missing values
        missing_data = prices.isnull().sum()
        total_rows = len(prices)
        
        for ticker, missing_count in missing_data.items():
            missing_pct = missing_count / total_rows * 100
            if missing_pct > 50:
                issues["errors"].append(f"{ticker}: More than 50% missing data ({missing_pct:.1f}%)")
            elif missing_pct > 10:
                issues["warnings"].append(f"{ticker}: High missing data ({missing_pct:.1f}%)")
                
        # Check for zero or negative prices
        zero_prices = (prices <= 0).sum()
        for ticker, zero_count in zero_prices.items():
            if zero_count > 0:
                issues["warnings"].append(f"{ticker}: {zero_count} zero/negative prices found")
                
        # Check for extreme price movements
        price_changes = prices.pct_change().abs()
        extreme_moves = (price_changes > 0.5).sum()
        for ticker, extreme_count in extreme_moves.items():
            if extreme_count > 0:
                issues["warnings"].append(f"{ticker}: {extreme_count} extreme price movements (>50%)")
                
        # Check for data gaps
        for ticker in prices.columns:
            ticker_data = prices[ticker].dropna()
            if len(ticker_data) > 1:
                # Only check for gaps if index is datetime
                if isinstance(ticker_data.index, pd.DatetimeIndex):
                    gaps = ticker_data.index.to_series().diff().dt.days > 7
                    gap_count = gaps.sum()
                    if gap_count > 0:
                        issues["warnings"].append(f"{ticker}: {gap_count} data gaps (>7 days) detected")
                    
        # Add data info
        issues["info"].append(f"Data spans {prices.index.min()} to {prices.index.max()}")
        issues["info"].append(f"Total observations: {len(prices)}")
        issues["info"].append(f"Number of tickers: {len(prices.columns)}")
        
        return issues
    
    @staticmethod
    def validate_returns_data(returns: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate returns data for common issues.
        
        Args:
            returns: DataFrame with date index and ticker columns
            
        Returns:
            Dictionary with validation results
        """
        issues = {
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        if returns.empty:
            issues["errors"].append("Returns data is empty")
            return issues
            
        # Check for extreme returns
        extreme_returns = (returns.abs() > 0.5).sum()
        for ticker, extreme_count in extreme_returns.items():
            if extreme_count > 0:
                issues["warnings"].append(f"{ticker}: {extreme_count} extreme returns (>50%)")
                
        # Check for constant returns (suspicious)
        for ticker in returns.columns:
            ticker_returns = returns[ticker].dropna()
            if len(ticker_returns) > 1:
                if ticker_returns.std() == 0:
                    issues["warnings"].append(f"{ticker}: Constant returns detected (suspicious)")
                    
        # Check for return statistics
        for ticker in returns.columns:
            ticker_returns = returns[ticker].dropna()
            if len(ticker_returns) > 0:
                mean_return = ticker_returns.mean()
                std_return = ticker_returns.std()
                
                if abs(mean_return) > 0.1:  # >10% daily return
                    issues["warnings"].append(f"{ticker}: High mean return ({mean_return:.3f})")
                    
                if std_return > 0.1:  # >10% daily volatility
                    issues["warnings"].append(f"{ticker}: High volatility ({std_return:.3f})")
                    
        return issues
    
    @staticmethod
    def check_data_consistency(prices: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, List[str]]:
        """Check consistency between price and returns data.
        
        Args:
            prices: Price DataFrame
            returns: Returns DataFrame
            
        Returns:
            Dictionary with consistency check results
        """
        issues = {
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check if indices match
        if not prices.index.equals(returns.index):
            issues["warnings"].append("Price and returns indices don't match")
            
        # Check if columns match
        price_tickers = set(prices.columns)
        returns_tickers = set(returns.columns)
        
        missing_in_returns = price_tickers - returns_tickers
        missing_in_prices = returns_tickers - price_tickers
        
        if missing_in_returns:
            issues["warnings"].append(f"Tickers in prices but not returns: {missing_in_returns}")
        if missing_in_prices:
            issues["warnings"].append(f"Tickers in returns but not prices: {missing_in_prices}")
            
        # Check for common tickers
        common_tickers = price_tickers & returns_tickers
        if common_tickers:
            issues["info"].append(f"Common tickers: {len(common_tickers)}")
            
        return issues
    
    @staticmethod
    def get_data_quality_score(prices: pd.DataFrame) -> float:
        """Calculate a data quality score (0-1).
        
        Args:
            prices: Price DataFrame
            
        Returns:
            Quality score between 0 and 1
        """
        if prices.empty:
            return 0.0
            
        score = 1.0
        
        # Penalize missing data
        missing_pct = prices.isnull().sum().sum() / (len(prices) * len(prices.columns))
        score -= missing_pct * 0.5  # Up to 50% penalty for missing data
        
        # Penalize zero/negative prices
        zero_pct = (prices <= 0).sum().sum() / (len(prices) * len(prices.columns))
        score -= zero_pct * 0.3  # Up to 30% penalty for zero prices
        
        # Penalize data gaps
        gap_penalty = 0
        for ticker in prices.columns:
            ticker_data = prices[ticker].dropna()
            if len(ticker_data) > 1:
                gaps = ticker_data.index.to_series().diff().dt.days > 7
                gap_penalty += gaps.sum() / len(ticker_data)
                
        gap_penalty = min(gap_penalty / len(prices.columns), 0.2)  # Up to 20% penalty
        score -= gap_penalty
        
        return max(0.0, score)
    
    @staticmethod
    def suggest_data_cleaning(prices: pd.DataFrame) -> Dict[str, List[str]]:
        """Suggest data cleaning operations.
        
        Args:
            prices: Price DataFrame
            
        Returns:
            Dictionary with cleaning suggestions
        """
        suggestions = {
            "required": [],
            "recommended": [],
            "optional": []
        }
        
        # Check for missing data
        missing_data = prices.isnull().sum()
        for ticker, missing_count in missing_data.items():
            missing_pct = missing_count / len(prices) * 100
            if missing_pct > 50:
                suggestions["required"].append(f"Remove {ticker} (too much missing data)")
            elif missing_pct > 10:
                suggestions["recommended"].append(f"Interpolate missing data for {ticker}")
                
        # Check for zero prices
        zero_prices = (prices <= 0).sum()
        for ticker, zero_count in zero_prices.items():
            if zero_count > 0:
                suggestions["required"].append(f"Handle zero/negative prices for {ticker}")
                
        # Check for extreme outliers
        for ticker in prices.columns:
            ticker_data = prices[ticker].dropna()
            if len(ticker_data) > 0:
                q1 = ticker_data.quantile(0.01)
                q99 = ticker_data.quantile(0.99)
                outliers = ((ticker_data < q1) | (ticker_data > q99)).sum()
                if outliers > 0:
                    suggestions["recommended"].append(f"Check outliers for {ticker} ({outliers} found)")
                    
        return suggestions
