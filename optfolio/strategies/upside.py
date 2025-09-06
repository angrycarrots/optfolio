
import pandas as pd
import numpy as np
from pathlib import Path



class UpsideCalculator:
    """Calculate analyst price target upside for stocks to use as Black-Litterman views."""
    
    def __init__(self, datadir: str = "data/upside/"):
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
        df = pd.read_csv(self.datadir / f"{symbol.upper()}.csv")
        return df
    
    def get_available_symbols(self) -> list:
        """Get the available symbols in the upside directory."""
        return list(self.datadir.glob("*.csv"))
    
if __name__ == "__main__":
    uc = UpsideCalculator()
    print(uc.get_available_symbols())
    print(uc.upside('WSM'))