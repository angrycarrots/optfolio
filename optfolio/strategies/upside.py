import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add the core directory to the path for imports
sys.path.append("../core")
from tools.scrape.stockanalysis_util import StockAnalysisContents, StockAnalysisUnpack, extract_key


class UpsideCalculator:
    """Calculate analyst price target upside for stocks to use as Black-Litterman views."""
    
    def __init__(self, datadir: str = "data/model/"):
        """Initialize the UpsideCalculator.
        
        Args:
            datadir: Path to the directory containing stock analysis JSON files
        """
        self.datadir = Path(datadir)
        
    def upside(self, symbol: str) -> pd.DataFrame:
        """Calculate the analyst price target upside for a given symbol as a function of date.
        
        The upside can then be used as a Black-Litterman view.
        
        Args:
            symbol: The symbol of the stock to calculate the upside for

        Returns:
            A DataFrame with columns: close, price_target, upside
            
        Example output:
                      close  price_target    upside
        date                                       
        2015-11-16   32.615          41.5  0.272421
        2015-11-17   33.370          41.5  0.243632
        2015-11-18   33.105          41.5  0.253587
        2015-11-19   31.970          41.5  0.298092
        2015-11-22   31.815          41.5  0.304416
        ...             ...           ...       ...
        2025-08-28  188.190         200.0  0.062756
        2025-09-01  189.400         215.0  0.135164
        """
        # Construct the file path
        json_file = self.datadir / f"{symbol.upper()}.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Data file not found for symbol {symbol}: {json_file}")
        
        # Load and parse the JSON data
        try:
            with open(json_file, 'r') as f:
                jdata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Error reading data file for {symbol}: {e}")
        
        # Extract price history
        pricedf = pd.DataFrame(jdata['price_history'])[['date', 'close']]
        pricedf.set_index('date', inplace=True)
        
        # Ensure close prices are numeric
        pricedf['close'] = pd.to_numeric(pricedf['close'], errors='coerce')
        
        # Extract ratings data
        X = StockAnalysisUnpack(jdata['ratings'])
        ddata = X.explode(0)
        df = pd.DataFrame(ddata['ratings'])
        df.set_index('date', inplace=True)
        
        # Join price data with price targets
        adf = pricedf.join(df[['pt_now']])
        
        # Ensure price targets are numeric and forward fill missing values
        adf['pt_now'] = pd.to_numeric(adf['pt_now'], errors='coerce')
        adf['price_target'] = adf['pt_now'].ffill()  # Use ffill() instead of deprecated fillna(method='ffill')
        adf['upside'] = (adf['price_target'] - adf['close']) / adf['close']
        
        # Remove any rows with missing data or infinite values
        adf = adf.dropna()
        adf = adf.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Ensure all data is finite
        adf = adf[np.isfinite(adf['upside'])]
        
        # Return only the relevant columns
        return adf[['close', 'price_target', 'upside']]
    
    def get_available_symbols(self) -> list:
        """Get a list of all available symbols in the data directory.
        
        Returns:
            List of available stock symbols
        """
        if not self.datadir.exists():
            return []
        
        json_files = list(self.datadir.glob("*.json"))
        symbols = [f.stem for f in json_files]
        return sorted(symbols)




if __name__ == "__main__":
    uc = UpsideCalculator()
    print(uc.get_available_symbols())
    print(uc.upside('WSM'))