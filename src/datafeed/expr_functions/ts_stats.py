"""
Time series statistics functions.

This module provides functions for computing various time series statistics.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional

def ts_correlation(x: pd.Series, y: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling correlation between two series.
    
    Args:
        x: First series
        y: Second series
        periods: Rolling window size
        
    Returns:
        Series of rolling correlations
    """
    # Ensure both series have the same index
    if not x.index.equals(y.index):
        raise ValueError("Series must have the same index")
    
    # Calculate rolling correlation
    return x.rolling(window=periods).corr(y)

def ts_covariance(x: pd.Series, y: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling covariance between two series.
    
    Args:
        x: First series
        y: Second series
        periods: Rolling window size
        
    Returns:
        Series of rolling covariances
    """
    # Ensure both series have the same index
    if not x.index.equals(y.index):
        raise ValueError("Series must have the same index")
    
    # Calculate rolling covariance
    return x.rolling(window=periods).cov(y)

def ts_skew(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling skewness.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling skewness values
    """
    return x.rolling(window=periods).skew()

def ts_kurt(x: pd.Series, periods: int = 10) -> pd.Series:
    """Calculate rolling kurtosis.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Series of rolling kurtosis values
    """
    return x.rolling(window=periods).kurt()

def ts_scale(x: pd.Series, periods: int = 10) -> pd.Series:
    """Scale time series to zero mean and unit variance.
    
    Args:
        x: Input series
        periods: Rolling window size
        
    Returns:
        Scaled series
    """
    mean = x.rolling(window=periods).mean()
    std = x.rolling(window=periods).std()
    return (x - mean) / std 