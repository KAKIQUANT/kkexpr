"""
Real data benchmarking using A-share market data.

This module demonstrates factor computation on real market data,
implementing a complex WorldQuant 101 alpha factor.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
import time

from datafeed.expr_functions import (
    ts_mean, ts_std, ts_delay, ts_delta, ts_rank,
    ts_pct_change, ts_max, ts_min, ts_correlation,
    ts_covariance, ts_scale
)
from datafeed.rust_expr import alpha101_factor_42 as rs_alpha42

def load_index_data(data_dir: str = "data/indexes") -> pd.DataFrame:
    """Load and stack all SH index data.
    
    Args:
        data_dir: Directory containing index data files
        
    Returns:
        DataFrame with stacked index data
    """
    # Find all SH index files
    pattern = os.path.join(data_dir, "*.SH.csv")
    files = glob.glob(pattern)
    
    dfs = []
    for file in files:
        # Extract symbol from filename
        symbol = os.path.basename(file).split('.')[0] + '.SH'
        
        # Read CSV
        df = pd.read_csv(file)
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['date'])
        dfs.append(df)
    
    # Stack all dataframes
    data = pd.concat(dfs, axis=0)
    
    # Set multi-index
    data = data.set_index(['date', 'symbol']).sort_index()
    
    return data

def alpha101_factor_42(data: pd.DataFrame) -> pd.Series:
    """WorldQuant 101 Alpha Factor #42.
    
    (-1 × rank(stddev(high, 10))) × correlation(high, volume, 10)
    
    This factor combines price volatility with volume-price correlation,
    capturing both volatility regimes and volume-price relationships.
    
    Args:
        data: DataFrame with high prices and volume
        
    Returns:
        Series with factor values
    """
    high = data['high']
    volume = data['volume']
    
    # Calculate components
    vol_rank = -1 * ts_rank(ts_std(high, periods=10), periods=10)
    vol_price_corr = ts_correlation(high, volume, periods=10)
    
    # Combine components
    return vol_rank * vol_price_corr

def alpha101_factor_42_rust(high: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Rust implementation of WorldQuant 101 Alpha Factor #42."""
    return rs_alpha42(high, volume)

def benchmark_implementation(
    func: Callable,
    data: pd.DataFrame,
    name: str,
    **kwargs
) -> Dict[str, Any]:
    """Benchmark a specific implementation.
    
    Args:
        func: Function to benchmark
        data: Input data
        name: Name of the implementation
        **kwargs: Additional arguments for the function
        
    Returns:
        Dictionary with timing results
    """
    # Convert data to appropriate format if needed
    if name.startswith('rust'):
        high_arr = data['high'].to_numpy(dtype=np.float64)
        volume_arr = data['volume'].to_numpy(dtype=np.float64)
        data_arr = (high_arr, volume_arr)
    else:
        data_arr = data
    
    # Warm-up run
    _ = func(*data_arr) if name.startswith('rust') else func(data_arr)
    
    # Timed runs
    times = []
    for _ in range(5):
        start_time = time.perf_counter()
        result = func(*data_arr) if name.startswith('rust') else func(data_arr)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result_shape': result.shape if hasattr(result, 'shape') else len(result)
    }

def run_comparison():
    """Run performance comparison between Python and Rust implementations."""
    print("Loading index data...")
    data = load_index_data()
    print(f"Dataset shape: {data.shape}")
    
    # Define test cases
    test_cases = [
        {
            'name': 'Alpha101 Factor #42',
            'implementations': {
                'python': (alpha101_factor_42, {}),
                'rust': (alpha101_factor_42_rust, {})
            }
        }
    ]
    
    # Run benchmarks
    results = {}
    for case in test_cases:
        print(f"\nBenchmarking {case['name']}...")
        case_results = {}
        
        for impl_name, (func, kwargs) in case['implementations'].items():
            print(f"\n{impl_name.title()} implementation:")
            result = benchmark_implementation(func, data, impl_name, **kwargs)
            case_results[impl_name] = result
            
            print(f"Mean time: {result['mean_time']:.4f} seconds")
            print(f"Std time:  {result['std_time']:.4f} seconds")
            print(f"Min time:  {result['min_time']:.4f} seconds")
            print(f"Max time:  {result['max_time']:.4f} seconds")
        
        # Calculate speedup
        if 'python' in case_results and 'rust' in case_results:
            speedup = case_results['python']['mean_time'] / case_results['rust']['mean_time']
            print(f"\nRust speedup: {speedup:.2f}x")
        
        results[case['name']] = case_results
    
    return results

if __name__ == '__main__':
    results = run_comparison() 