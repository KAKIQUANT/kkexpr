"""
Example implementations of volatility-based factors.

This module demonstrates how to use the expression functions to create
volatility-based factors used in risk management and trading strategies.
"""

import pandas as pd
import numpy as np
from src.datafeed.expr_functions import (
    ts_std, ts_mean, ts_delay, ts_delta,
    ts_max, ts_min, ts_skew, ts_kurt
)

def create_sample_data():
    """Create sample price data for demonstration."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Create multi-index with dates and symbols
    index = pd.MultiIndex.from_product(
        [dates, symbols], 
        names=['date', 'symbol']
    )
    
    # Generate random price movements with varying volatility
    np.random.seed(42)
    n = len(index)
    
    # Create base returns with time-varying volatility
    base_vol = np.sin(np.linspace(0, 8*np.pi, n)) * 0.01 + 0.02
    returns = np.random.normal(0.0005, base_vol)
    prices = 100 * (1 + returns).cumprod()
    
    return pd.Series(prices, index=index, name='close')

def realized_volatility(prices: pd.Series, periods: int = 20) -> pd.Series:
    """
    Calculate realized volatility.
    
    Args:
        prices: Series of asset prices
        periods: Lookback period in days (default: 20)
        
    Returns:
        Series of annualized volatility values
    """
    # Calculate returns
    returns = ts_delta(prices, periods=1) / ts_delay(prices, periods=1)
    
    # Calculate annualized volatility
    vol = ts_std(returns, periods=periods) * np.sqrt(252)
    return vol

def volatility_regime(prices: pd.Series, slow: int = 252, fast: int = 20) -> pd.Series:
    """
    Identify volatility regime using ratio of short-term to long-term volatility.
    
    Args:
        prices: Series of asset prices
        slow: Long-term lookback period (default: 252)
        fast: Short-term lookback period (default: 20)
        
    Returns:
        Series of volatility regime indicators
    """
    vol_fast = realized_volatility(prices, periods=fast)
    vol_slow = realized_volatility(prices, periods=slow)
    
    return vol_fast / vol_slow

def parkinson_volatility(data: pd.DataFrame, periods: int = 20) -> pd.Series:
    """
    Calculate Parkinson volatility using high-low range.
    
    Args:
        data: DataFrame with high and low prices
        periods: Lookback period (default: 20)
        
    Returns:
        Series of Parkinson volatility estimates
    """
    # Calculate normalized high-low range
    hl_range = np.log(data['high'] / data['low'])
    range_sq = hl_range ** 2
    
    # Calculate Parkinson volatility (annualized)
    factor = 1 / (4 * np.log(2))
    return np.sqrt(factor * ts_mean(range_sq, periods=periods) * 252)

def tail_risk_factor(prices: pd.Series, periods: int = 60) -> pd.DataFrame:
    """
    Calculate tail risk measures using higher moments.
    
    Args:
        prices: Series of asset prices
        periods: Lookback period (default: 60)
        
    Returns:
        DataFrame with skewness and kurtosis measures
    """
    # Calculate returns
    returns = ts_delta(prices, periods=1) / ts_delay(prices, periods=1)
    
    # Calculate rolling higher moments
    skewness = ts_skew(returns, periods=periods)
    kurtosis = ts_kurt(returns, periods=periods)
    
    return pd.DataFrame({
        'skewness': skewness,
        'kurtosis': kurtosis
    }, index=returns.index)

def volatility_breakout(prices: pd.Series, periods: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Identify volatility breakouts.
    
    Args:
        prices: Series of asset prices
        periods: Lookback period (default: 20)
        num_std: Number of standard deviations for breakout (default: 2.0)
        
    Returns:
        Series of volatility breakout signals
    """
    # Calculate rolling volatility
    vol = realized_volatility(prices, periods=periods)
    
    # Calculate volatility of volatility
    vol_mean = ts_mean(vol, periods=periods)
    vol_std = ts_std(vol, periods=periods)
    
    # Calculate z-score of current volatility
    return (vol - vol_mean) / vol_std

def main():
    """Run example volatility factor calculations."""
    # Create sample data
    prices = create_sample_data()
    
    # Calculate factors
    vol = realized_volatility(prices)
    regime = volatility_regime(prices)
    tail_risk = tail_risk_factor(prices)
    vol_breakout = volatility_breakout(prices)
    
    # Display results for the last date
    last_date = prices.index.get_level_values('date').max()
    print(f"\nVolatility measures for {last_date.strftime('%Y-%m-%d')}:")
    
    print("\nRealized Volatility (annualized):")
    print(vol.xs(last_date, level='date').round(4))
    
    print("\nVolatility Regime (fast/slow ratio):")
    print(regime.xs(last_date, level='date').round(4))
    
    print("\nTail Risk Measures:")
    print(tail_risk.xs(last_date, level='date').round(4))
    
    print("\nVolatility Breakout Signals (z-score):")
    print(vol_breakout.xs(last_date, level='date').round(4))

if __name__ == '__main__':
    main() 