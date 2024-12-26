"""
Example implementations of momentum-based factors.

This module demonstrates how to use the expression functions to create
common momentum factors used in quantitative trading strategies.
"""

import pandas as pd
import numpy as np
from datafeed.expr_functions import (
    ts_mean, ts_std, ts_delay, ts_delta, ts_rank,
    ts_pct_change, ts_max, ts_min, ts_maxmin
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
    
    # Generate random price movements
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(index))
    prices = 100 * (1 + returns).cumprod()
    
    return pd.Series(prices, index=index, name='close')

def momentum_factor(prices: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Classic price momentum factor.
    
    Calculates the momentum as the percentage change over a lookback period,
    normalized by volatility.
    
    Args:
        prices: Series of asset prices
        lookback: Lookback period in days (default: 252 for 1 year)
        
    Returns:
        Series of momentum scores
    """
    # Calculate returns
    returns = ts_pct_change(prices, periods=1)
    
    # Calculate momentum (return over lookback period)
    momentum = ts_pct_change(prices, periods=lookback)
    
    # Calculate volatility
    vol = ts_std(returns, periods=lookback)
    
    # Normalize momentum by volatility
    return momentum / vol

def mean_reversion_factor(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Mean reversion factor based on z-score of price.
    
    Calculates how many standard deviations the current price is away
    from its moving average.
    
    Args:
        prices: Series of asset prices
        lookback: Lookback period in days (default: 20 for 1 month)
        
    Returns:
        Series of mean reversion scores
    """
    # Calculate moving average
    ma = ts_mean(prices, periods=lookback)
    
    # Calculate standard deviation
    std = ts_std(prices, periods=lookback)
    
    # Calculate z-score
    return -(prices - ma) / std  # Negative sign for mean reversion

def breakout_factor(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Breakout strength factor.
    
    Measures how close the current price is to its recent high,
    normalized by the high-low range.
    
    Args:
        prices: Series of asset prices
        lookback: Lookback period in days (default: 20 for 1 month)
        
    Returns:
        Series of breakout scores
    """
    return ts_maxmin(prices, periods=lookback)

def relative_strength_factor(prices: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Relative strength factor based on rolling ranks.
    
    Combines multiple timeframe momentum signals into a composite score.
    
    Args:
        prices: Series of asset prices
        lookback: Base lookback period in days (default: 60)
        
    Returns:
        Series of relative strength scores
    """
    # Calculate momentum over different timeframes
    mom_st = ts_pct_change(prices, periods=lookback // 3)  # Short-term
    mom_mt = ts_pct_change(prices, periods=lookback)       # Medium-term
    mom_lt = ts_pct_change(prices, periods=lookback * 2)   # Long-term
    
    # Rank each momentum signal
    rank_st = ts_rank(mom_st, periods=lookback)
    rank_mt = ts_rank(mom_mt, periods=lookback)
    rank_lt = ts_rank(mom_lt, periods=lookback)
    
    # Combine ranks with decreasing weights
    return (0.5 * rank_st + 0.3 * rank_mt + 0.2 * rank_lt)

def main():
    """Run example factor calculations."""
    # Create sample data
    prices = create_sample_data()
    
    # Calculate factors
    factors = {
        'momentum': momentum_factor(prices),
        'mean_reversion': mean_reversion_factor(prices),
        'breakout': breakout_factor(prices),
        'relative_strength': relative_strength_factor(prices)
    }
    
    # Display results for the last date
    last_date = prices.index.get_level_values('date').max()
    print(f"\nFactor values for {last_date.strftime('%Y-%m-%d')}:")
    for name, factor in factors.items():
        print(f"\n{name.title()} Factor:")
        print(factor.xs(last_date, level='date').round(4))

if __name__ == '__main__':
    main() 