"""
Time series unary functions.

This module provides functions for computing various unary time series operations.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional

def ts_mean(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling mean.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling means
    """
    return x.rolling(window=periods).mean()

def ts_std(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling standard deviation.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling standard deviations
    """
    return x.rolling(window=periods).std()

def ts_delay(x: pd.Series, periods: int = 1) -> pd.Series:
    """Delay (lag) series by specified number of periods.
    
    Args:
        x: Input series
        periods: Number of periods to lag
        
    Returns:
        Lagged series
    """
    return x.shift(periods)

def ts_delta(x: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate difference between current value and lagged value.
    
    Args:
        x: Input series
        periods: Number of periods to lag
        
    Returns:
        Series of differences
    """
    return x - x.shift(periods)

def ts_pct_change(x: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage change between current value and lagged value.
    
    Args:
        x: Input series
        periods: Number of periods to lag
        
    Returns:
        Series of percentage changes
    """
    return x.pct_change(periods)

def ts_max(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling maximum.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling maximums
    """
    return x.rolling(window=periods).max()

def ts_min(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling minimum.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling minimums
    """
    return x.rolling(window=periods).min()

def ts_maxmin(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling max-min range.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling ranges
    """
    return ts_max(x, periods) - ts_min(x, periods)

def ts_rank(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling rank (percentile).
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling ranks
    """
    def rolling_rank(window):
        if len(window) < periods:
            return np.nan
        return (window.argsort().argsort()[-1] + 1) / len(window)
    
    return x.rolling(window=periods).apply(rolling_rank, raw=True) 