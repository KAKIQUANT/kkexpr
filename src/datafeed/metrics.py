"""Financial metrics calculation module."""

import numpy as np
import pandas as pd
from typing import Union, Optional

def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate the maximum drawdown of a return series.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as a positive number or np.nan for empty input
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
        
    if len(returns) == 0:
        return np.nan
        
    cumulative = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / rolling_max - 1
    return abs(float(np.min(drawdowns)))

def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free: Union[float, pd.Series, np.ndarray] = 0.0,
    periods: int = 252
) -> float:
    """Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free: Risk-free rate or series
        periods: Number of periods in a year (252 for daily data)
        
    Returns:
        Annualized Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(risk_free, pd.Series):
        risk_free = risk_free.values
        
    excess_returns = returns - risk_free
    if not len(excess_returns):
        return np.nan
        
    mean = np.mean(excess_returns)
    std = np.std(excess_returns, ddof=1)
    
    if std == 0:
        return np.nan
        
    return np.sqrt(periods) * mean / std

def annual_return(
    returns: Union[pd.Series, np.ndarray],
    periods: int = 252
) -> float:
    """Calculate annualized return.
    
    Args:
        returns: Series of returns
        periods: Number of periods in a year (252 for daily data)
        
    Returns:
        Annualized return
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
        
    if not len(returns):
        return np.nan
        
    cumulative = (1 + returns).prod()
    years = len(returns) / periods
    return cumulative ** (1 / years) - 1

def sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free: Union[float, pd.Series, np.ndarray] = 0.0,
    periods: int = 252
) -> float:
    """Calculate the annualized Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free: Risk-free rate or series
        periods: Number of periods in a year (252 for daily data)
        
    Returns:
        Annualized Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(risk_free, pd.Series):
        risk_free = risk_free.values
        
    excess_returns = returns - risk_free
    if not len(excess_returns):
        return np.nan
        
    mean = np.mean(excess_returns)
    downside = excess_returns[excess_returns < 0]
    
    if not len(downside):
        return np.inf if mean > 0 else np.nan
        
    downside_std = np.std(downside, ddof=1)
    
    if downside_std == 0:
        return np.nan
        
    return np.sqrt(periods) * mean / downside_std

def calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods: int = 252
) -> float:
    """Calculate Calmar ratio.
    
    Args:
        returns: Series of returns
        periods: Number of periods in a year (252 for daily data)
        
    Returns:
        Calmar ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
        
    if not len(returns):
        return np.nan
        
    ann_return = annual_return(returns, periods)
    max_dd = max_drawdown(returns)
    
    if max_dd == 0:
        return np.nan
        
    return ann_return / max_dd 