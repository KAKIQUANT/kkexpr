"""
Example implementations of technical indicators.

This module demonstrates how to use the expression functions to create
common technical indicators used in trading strategies.
"""

import pandas as pd
import numpy as np
from src.datafeed.expr_functions import (
    ts_mean, ts_std, ts_max, ts_min, ts_sum,
    ts_pct_change, ts_delay, ts_delta
)

def create_sample_data():
    """Create sample OHLCV data for demonstration."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Create multi-index with dates and symbols
    index = pd.MultiIndex.from_product(
        [dates, symbols], 
        names=['date', 'symbol']
    )
    
    # Generate random price movements
    np.random.seed(42)
    n = len(index)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': 100 * (1 + np.random.normal(0.0005, 0.02, n)).cumprod(),
        'high': 100 * (1 + np.random.normal(0.001, 0.02, n)).cumprod(),
        'low': 100 * (1 + np.random.normal(0, 0.02, n)).cumprod(),
        'close': 100 * (1 + np.random.normal(0.0005, 0.02, n)).cumprod(),
        'volume': np.random.lognormal(10, 1, n)
    }, index=index)
    
    # Adjust high/low to be consistent
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def bollinger_bands(prices: pd.Series, periods: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of asset prices
        periods: Lookback period for moving average (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
        
    Returns:
        DataFrame with middle, upper, and lower bands
    """
    middle = ts_mean(prices, periods=periods)
    std = ts_std(prices, periods=periods)
    
    return pd.DataFrame({
        'middle': middle,
        'upper': middle + (std * num_std),
        'lower': middle - (std * num_std)
    }, index=prices.index)

def rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of asset prices
        periods: Lookback period (default: 14)
        
    Returns:
        Series of RSI values
    """
    # Calculate price changes
    changes = ts_delta(prices, periods=1)
    
    # Separate gains and losses
    gains = changes.copy()
    gains[gains < 0] = 0
    
    losses = -changes.copy()
    losses[losses < 0] = 0
    
    # Calculate average gains and losses
    avg_gains = ts_mean(gains, periods=periods)
    avg_losses = ts_mean(losses, periods=periods)
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    return 100 - (100 / (1 + rs))

def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Series of asset prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
        
    Returns:
        DataFrame with MACD line, signal line, and histogram
    """
    # Calculate EMAs
    ema_fast = ts_mean(prices, periods=fast)
    ema_slow = ts_mean(prices, periods=slow)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = ts_mean(macd_line, periods=signal)
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': macd_line - signal_line
    }, index=prices.index)

def atr(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        data: DataFrame with high, low, and close prices
        periods: Lookback period (default: 14)
        
    Returns:
        Series of ATR values
    """
    # Calculate true range
    prev_close = ts_delay(data['close'], periods=1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    return ts_mean(true_range, periods=periods)

def main():
    """Run example technical indicator calculations."""
    # Create sample data
    data = create_sample_data()
    
    # Calculate indicators for the last symbol
    last_symbol = data.index.get_level_values('symbol')[-1]
    symbol_data = data.xs(last_symbol, level='symbol')
    
    # Calculate indicators
    bb = bollinger_bands(symbol_data['close'])
    rsi_values = rsi(symbol_data['close'])
    macd_values = macd(symbol_data['close'])
    atr_values = atr(symbol_data)
    
    # Display results for the last date
    last_date = data.index.get_level_values('date').max()
    print(f"\nIndicator values for {last_symbol} on {last_date.strftime('%Y-%m-%d')}:")
    
    print("\nBollinger Bands:")
    print(bb.loc[last_date].round(2))
    
    print(f"\nRSI: {rsi_values.loc[last_date]:.2f}")
    
    print("\nMACD:")
    print(macd_values.loc[last_date].round(4))
    
    print(f"\nATR: {atr_values.loc[last_date]:.4f}")

if __name__ == '__main__':
    main() 