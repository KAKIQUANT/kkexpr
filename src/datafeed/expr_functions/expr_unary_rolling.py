import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Union, Tuple
import ta

from .expr_utils import calc_by_symbol
from .expr_exceptions import InvalidPeriodError
from .expr_config import config

def validate_period(periods: int, func_name: str) -> None:
    """Validate period parameter against configuration.
    
    Args:
        periods: Number of periods
        func_name: Name of the function for config lookup
        
    Raises:
        InvalidPeriodError: If period is invalid
    """
    if periods < config.min_periods.get(func_name, 1):
        raise InvalidPeriodError(
            f"Period must be >= {config.min_periods.get(func_name, 1)} for {func_name}"
        )

@calc_by_symbol
def ts_delay(se: pd.Series, periods: int = config.default_periods['delay']) -> pd.Series:
    """Delay a time series by N periods.
    
    Args:
        se: Input time series
        periods: Number of periods to delay
        
    Returns:
        Series shifted by specified periods
        
    Raises:
        InvalidPeriodError: If periods is invalid
    """
    validate_period(periods, 'delay')
    return se.shift(periods=periods)

@calc_by_symbol
def ts_delta(se: pd.Series, periods: int = config.default_periods['delta']) -> pd.Series:
    """Calculate difference between current value and N periods ago.
    
    Args:
        se: Input time series
        periods: Number of periods for difference
        
    Returns:
        Series of differences
    """
    validate_period(periods, 'delta')
    return se - se.shift(periods=periods)

@calc_by_symbol
def ts_mean(se: pd.Series, periods: int = config.default_periods['mean']) -> pd.Series:
    """Calculate rolling mean.
    
    Args:
        se: Input time series
        periods: Window size for rolling mean
        
    Returns:
        Series of rolling means
    """
    validate_period(periods, 'mean')
    return se.rolling(window=periods).mean()

@calc_by_symbol
def ts_median(se: pd.Series, periods: int = config.default_periods['median']) -> pd.Series:
    """Calculate rolling median.
    
    Args:
        se: Input time series
        periods: Window size for rolling median
        
    Returns:
        Series of rolling medians
    """
    validate_period(periods, 'median')
    return se.rolling(window=periods).median()

@calc_by_symbol
def ts_pct_change(se: pd.Series, periods: int = config.default_periods['delta']) -> pd.Series:
    """Calculate percentage change over N periods.
    
    Args:
        se: Input time series
        periods: Number of periods for change calculation
        
    Returns:
        Series of percentage changes
    """
    validate_period(periods, 'delta')
    return se / se.shift(periods) - 1

@calc_by_symbol
def ts_max(se: pd.Series, periods: int = config.default_periods['max']) -> pd.Series:
    """Calculate rolling maximum.
    
    Args:
        se: Input time series
        periods: Window size for rolling maximum
        
    Returns:
        Series of rolling maximums
    """
    validate_period(periods, 'max')
    return se.rolling(window=periods).max()

@calc_by_symbol
def ts_min(se: pd.Series, periods: int = config.default_periods['min']) -> pd.Series:
    """Calculate rolling minimum.
    
    Args:
        se: Input time series
        periods: Window size for rolling minimum
        
    Returns:
        Series of rolling minimums
    """
    validate_period(periods, 'min')
    return se.rolling(window=periods).min()

@calc_by_symbol
def ts_maxmin(se: pd.Series, periods: int = config.default_periods['max']) -> pd.Series:
    """Calculate normalized position within min-max range.
    
    Args:
        se: Input time series
        periods: Window size for min-max calculation
        
    Returns:
        Series of normalized values between 0 and 1
    """
    validate_period(periods, 'max')
    min_val = ts_min(se, periods)
    max_val = ts_max(se, periods)
    return (se - min_val) / (max_val - min_val)

@calc_by_symbol
def ts_sum(se: pd.Series, periods: int = config.default_periods['mean']) -> pd.Series:
    """Calculate rolling sum.
    
    Args:
        se: Input time series
        periods: Window size for rolling sum
        
    Returns:
        Series of rolling sums
    """
    validate_period(periods, 'mean')
    return se.rolling(window=periods).sum()

@calc_by_symbol
def ts_std(se: pd.Series, periods: int = config.default_periods['std']) -> pd.Series:
    """Calculate rolling standard deviation.
    
    Args:
        se: Input time series
        periods: Window size for rolling std
        
    Returns:
        Series of rolling standard deviations
    """
    validate_period(periods, 'std')
    return se.rolling(window=periods).std()

@calc_by_symbol
def ts_skew(se: pd.Series, periods: int = config.default_periods['skew']) -> pd.Series:
    """Calculate rolling skewness.
    
    Args:
        se: Input time series
        periods: Window size for rolling skewness
        
    Returns:
        Series of rolling skewness values
    """
    validate_period(periods, 'skew')
    return se.rolling(window=periods).skew()

@calc_by_symbol
def ts_kurt(se: pd.Series, periods: int = config.default_periods['kurt']) -> pd.Series:
    """Calculate rolling kurtosis.
    
    Args:
        se: Input time series
        periods: Window size for rolling kurtosis
        
    Returns:
        Series of rolling kurtosis values
    """
    validate_period(periods, 'kurt')
    return se.rolling(window=periods).kurt()

@calc_by_symbol
def ts_argmin(se: pd.Series, periods: int = config.default_periods['argmin']) -> pd.Series:
    """Calculate rolling argmin (position of minimum value).
    
    Args:
        se: Input time series
        periods: Window size for rolling argmin
        
    Returns:
        Series of positions of minimum values
    """
    validate_period(periods, 'argmin')
    return se.rolling(periods, min_periods=1).apply(lambda x: x.argmin())

@calc_by_symbol
def ts_argmax(se: pd.Series, periods: int = config.default_periods['argmax']) -> pd.Series:
    """Calculate rolling argmax (position of maximum value).
    
    Args:
        se: Input time series
        periods: Window size for rolling argmax
        
    Returns:
        Series of positions of maximum values
    """
    validate_period(periods, 'argmax')
    return se.rolling(periods, min_periods=1).apply(lambda x: x.argmax())

@calc_by_symbol
def ts_argmaxmin(se: pd.Series, periods: int = config.default_periods['argmax']) -> pd.Series:
    """Calculate difference between positions of max and min values.
    
    Args:
        se: Input time series
        periods: Window size for calculation
        
    Returns:
        Series of differences between max and min positions
    """
    validate_period(periods, 'argmax')
    return ts_argmax(se, periods) - ts_argmin(se, periods)

@calc_by_symbol
def ts_rank(se: pd.Series, periods: int = config.default_periods['rank']) -> pd.Series:
    """Calculate rolling rank (percentile).
    
    Args:
        se: Input time series
        periods: Window size for rolling rank
        
    Returns:
        Series of rolling ranks (0-1)
    """
    validate_period(periods, 'rank')
    
    def rank_pct(x):
        """Calculate percentile rank for the last value in window."""
        if len(x) < periods:
            return np.nan
        return x.rank(pct=True).iloc[-1]
    
    return se.rolling(window=periods, min_periods=periods).apply(rank_pct)

@calc_by_symbol
def ma(se: pd.Series, periods: int = config.default_periods['mean']) -> pd.Series:
    """Calculate simple moving average.
    
    Args:
        se: Input time series
        periods: Window size for moving average
        
    Returns:
        Series of moving averages
    """
    validate_period(periods, 'mean')
    se.ffill(inplace=True)
    return se.rolling(window=periods).mean()

