import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm
from loguru import logger

from config import DATA_DIR
from datafeed.expr import calc_expr

class DataLoader(ABC):
    """Abstract base class for data loading implementations."""
    
    @abstractmethod
    def read_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Read data for a single symbol.
        
        Args:
            symbol: The symbol/ticker to load
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            DataFrame containing the data or None if not found
        """
        pass
    
    @abstractmethod
    def get_df(self, 
               symbols: Optional[List[str]] = None,
               start_date: str = '20100101',
               end_date: Optional[str] = None,
               set_index: bool = False,
               **kwargs) -> pd.DataFrame:
        """Get a DataFrame containing data for multiple symbols.
        
        Args:
            symbols: List of symbols to load. If None, loads all available symbols
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            set_index: Whether to set date as index
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            DataFrame containing the combined data
        """
        pass

    def calc_expr(self, df: pd.DataFrame, fields: List[str], names: List[str]) -> pd.DataFrame:
        """Calculate expressions on the data.
        
        Args:
            df: Input DataFrame
            fields: List of expressions to calculate
            names: Names for the calculated fields
            
        Returns:
            DataFrame with calculated expressions
        """
        cols = []
        count = 0
        df.set_index([df.index, 'symbol'], inplace=True)
        
        for field, name in tqdm(zip(fields, names)):
            try:
                if not field:
                    continue
                se = calc_expr(df, field)
                
                count += 1
                if count < 10:
                    df[name] = se
                else:
                    se.name = name
                    cols.append(se)
            except Exception as e:
                logger.error(f'Error calculating expression {field}: {str(e)}')
                continue
                
        if cols:
            df_cols = pd.concat(cols, axis=1)
            df = pd.concat([df, df_cols], axis=1)

        df['symbol'] = df.index.droplevel(0)
        df.index = df.index.droplevel(1)
        return df

    def get_col_df(self,
                   df: pd.DataFrame,
                   col: str = 'close',
                   start_date: str = '20100101',
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """Pivot a column into a symbol-wise DataFrame.
        
        Args:
            df: Input DataFrame
            col: Column to pivot
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Pivoted DataFrame with symbols as columns
        """
        if col not in df.columns:
            logger.error(f'Column {col} does not exist')
            return pd.DataFrame()
            
        df_pivot = df.pivot_table(values=col, index=df.index, columns='symbol', dropna=False)
        df_pivot.ffill(inplace=True)
        
        df_pivot = df_pivot[start_date:]
        if end_date:
            df_pivot = df_pivot[:end_date]
            
        if isinstance(df_pivot, pd.Series):
            df_pivot = df_pivot.to_frame()
            
        df_pivot.fillna(0, inplace=True)
        return df_pivot

class CSVDataLoader(DataLoader):
    """Data loader implementation for CSV files."""
    
    def __init__(self, data_dir: Union[str, Path] = DATA_DIR):
        self.data_dir = Path(data_dir)
    
    def read_data(self, symbol: str, path: str = 'quotes') -> Optional[pd.DataFrame]:
        """Read data for a single symbol from CSV.
        
        Args:
            symbol: Symbol to load
            path: Subdirectory under data_dir containing the CSV files
            
        Returns:
            DataFrame containing the data or None if file not found
        """
        csv_path = self.data_dir.joinpath(path).joinpath(f'{symbol}.csv')
        if not csv_path.exists():
            logger.warning(f'File does not exist: {csv_path.resolve()}')
            return None

        df = pd.read_csv(csv_path.resolve(), index_col=None)
        df['date'] = pd.to_datetime(df['date'].astype(str))
        df['symbol'] = symbol
        df.dropna(inplace=True)
        return df
    
    def get_df(self,
               symbols: Optional[List[str]] = None,
               start_date: str = '20100101',
               end_date: Optional[str] = None,
               set_index: bool = False,
               path: str = 'quotes') -> pd.DataFrame:
        """Get data for multiple symbols from CSV files.
        
        Args:
            symbols: List of symbols to load. If None, loads all available symbols
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format (defaults to today)
            set_index: Whether to set date as index
            path: Subdirectory under data_dir containing the CSV files
            
        Returns:
            DataFrame containing the combined data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        if symbols is None:
            csv_dir = self.data_dir.joinpath(path)
            symbols = [p.stem for p in csv_dir.glob('*.csv')]
            
        dfs = []
        for symbol in symbols:
            df = self.read_data(symbol, path=path)
            if df is not None:
                dfs.append(df)
                
        if not dfs:
            return pd.DataFrame()
            
        df = pd.concat(dfs, axis=0)
        
        if set_index:
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True, ascending=True)
            df = df[start_date:end_date]
        else:
            df.sort_values(by='date', ascending=True, inplace=True)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
        return df
    
    def get_backtrader_df(self,
                         symbol: str,
                         start_date: str = '20050101',
                         end_date: Optional[str] = None,
                         path: str = 'quotes') -> pd.DataFrame:
        """Get data in Backtrader-compatible format.
        
        Args:
            symbol: Symbol to load
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            path: Subdirectory under data_dir containing the CSV files
            
        Returns:
            DataFrame formatted for Backtrader
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        df = self.get_df([symbol], start_date=start_date, path=path)
        df.set_index('date', inplace=True)
        df['openinterest'] = 0
        df = df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]
        df = df[df.index.astype(str) >= start_date]
        df = df[df.index.astype(str) <= end_date]
        return df
    
    @staticmethod
    def get_symbols_from_instruments(filename: str, data_dir: Union[str, Path] = DATA_DIR) -> List[str]:
        """Read symbol list from an instruments file.
        
        Args:
            filename: Name of the instruments file
            data_dir: Base data directory
            
        Returns:
            List of symbols
        """
        path = Path(data_dir).joinpath('instruments').joinpath(filename)
        with open(path.resolve(), 'r') as f:
            symbols = [line.strip() for line in f]
        return symbols


if __name__ == '__main__':
    import pandas as pd
    from datafeed.dataloader import CSVDataLoader

    symbols = CSVDataLoader.get_symbols_from_instruments('1-商品期货流动性好的品类.txt')
    #symbols = ['B0','BR0']

    df = CSVDataLoader.get_df(set_index=True, symbols=symbols)
    print(df)
    expr = 'RSRS(low,high,18)'
    df = CSVDataLoader.calc_expr(df, [expr,'slope_pair(high,low,18)'],
                                 ['factor','slope_2'])
    print(df)
