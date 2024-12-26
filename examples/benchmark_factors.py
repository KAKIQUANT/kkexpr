"""
Benchmark module for comparing factor computation performance.

This module creates a large dataset and measures the performance of
various factor computations.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
from datafeed.expr_functions import (
    ts_mean, ts_std, ts_delay, ts_delta, ts_rank,
    ts_pct_change, ts_max, ts_min, ts_maxmin,
    ts_skew, ts_kurt
)

def create_large_dataset(num_symbols: int = 1000) -> pd.DataFrame:
    """Create a large dataset for benchmarking.
    
    Args:
        num_symbols: Number of symbols to generate
        
    Returns:
        DataFrame with price data
    """
    # Generate dates from 2010 to 2023
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
    symbols = [f'STOCK_{i:04d}' for i in range(num_symbols)]
    
    # Create multi-index with dates and symbols
    index = pd.MultiIndex.from_product(
        [dates, symbols], 
        names=['date', 'symbol']
    )
    
    # Generate random price movements
    np.random.seed(42)
    n_dates = len(dates)
    n_symbols = len(symbols)
    
    # Generate systematic and idiosyncratic returns
    systematic_factor = np.random.normal(0.0005, 0.01, (n_dates, 1))
    idio_returns = np.random.normal(0, 0.02, (n_dates, n_symbols))
    
    # Combine returns
    returns = systematic_factor + idio_returns
    prices = 100 * np.cumprod(1 + returns, axis=0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'close': prices.flatten(),
        'volume': np.random.lognormal(10, 1, n_dates * n_symbols)
    }, index=index)
    
    return data

def benchmark_function(func: Callable, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        data: Input data
        **kwargs: Additional arguments for the function
        
    Returns:
        Dictionary with timing results
    """
    start_time = time.perf_counter()
    result = func(data['close'], **kwargs)
    end_time = time.perf_counter()
    
    return {
        'execution_time': end_time - start_time,
        'result_shape': result.shape,
        'memory_usage': result.memory_usage() / 1024 / 1024  # MB
    }

def run_benchmarks():
    """Run performance benchmarks on various factor computations."""
    print("Creating large dataset...")
    data = create_large_dataset()
    print(f"Dataset shape: {data.shape}")
    
    # Define benchmark cases
    benchmark_cases = {
        'Momentum (252-day)': lambda x: momentum_factor(x, lookback=252),
        'Mean Reversion (20-day)': lambda x: mean_reversion_factor(x, lookback=20),
        'Volatility (20-day)': lambda x: realized_volatility(x, periods=20),
        'Relative Strength (60-day)': lambda x: relative_strength_factor(x, lookback=60),
        'Tail Risk (60-day)': lambda x: tail_risk_factor(x, periods=60)
    }
    
    # Run benchmarks
    results = {}
    print("\nRunning benchmarks...")
    for name, func in benchmark_cases.items():
        print(f"\nBenchmarking {name}...")
        results[name] = benchmark_function(func, data)
        
        print(f"Execution time: {results[name]['execution_time']:.2f} seconds")
        print(f"Result shape: {results[name]['result_shape']}")
        print(f"Memory usage: {results[name]['memory_usage']:.2f} MB")
    
    return results

def momentum_factor(prices: pd.Series, lookback: int = 252) -> pd.Series:
    """Intensive momentum calculation."""
    returns = ts_pct_change(prices, periods=1)
    momentum = ts_pct_change(prices, periods=lookback)
    vol = ts_std(returns, periods=lookback)
    return momentum / vol

def mean_reversion_factor(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Intensive mean reversion calculation."""
    ma = ts_mean(prices, periods=lookback)
    std = ts_std(prices, periods=lookback)
    return -(prices - ma) / std

def realized_volatility(prices: pd.Series, periods: int = 20) -> pd.Series:
    """Intensive volatility calculation."""
    returns = ts_delta(prices, periods=1) / ts_delay(prices, periods=1)
    vol = ts_std(returns, periods=periods) * np.sqrt(252)
    return vol

def relative_strength_factor(prices: pd.Series, lookback: int = 60) -> pd.Series:
    """Intensive relative strength calculation."""
    # Multiple timeframe momentum
    timeframes = [lookback // 3, lookback, lookback * 2]
    momentum_signals = []
    
    for tf in timeframes:
        mom = ts_pct_change(prices, periods=tf)
        rank = ts_rank(mom, periods=lookback)
        momentum_signals.append(rank)
    
    # Combine with decreasing weights
    weights = np.array([0.5, 0.3, 0.2])
    return sum(w * sig for w, sig in zip(weights, momentum_signals))

def tail_risk_factor(prices: pd.Series, periods: int = 60) -> pd.Series:
    """Intensive tail risk calculation."""
    returns = ts_delta(prices, periods=1) / ts_delay(prices, periods=1)
    skewness = ts_skew(returns, periods=periods)
    kurtosis = ts_kurt(returns, periods=periods)
    
    # Combine metrics into a single score
    z_skew = (skewness - ts_mean(skewness, periods=periods)) / ts_std(skewness, periods=periods)
    z_kurt = (kurtosis - ts_mean(kurtosis, periods=periods)) / ts_std(kurtosis, periods=periods)
    
    return -z_skew * 0.5 - z_kurt * 0.5

if __name__ == '__main__':
    results = run_benchmarks() 