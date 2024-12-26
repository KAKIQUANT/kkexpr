"""
Compare performance between Python and Rust implementations.

This module runs benchmarks to compare the performance of factor calculations
between Python and Rust implementations.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
from benchmark_factors import (
    create_large_dataset,
    momentum_factor as py_momentum,
    mean_reversion_factor as py_mean_reversion,
    relative_strength_factor as py_relative_strength
)
from datafeed.rust_expr import (
    momentum_factor as rs_momentum,
    mean_reversion_factor as rs_mean_reversion,
    relative_strength_factor as rs_relative_strength
)

def benchmark_implementation(
    func: Callable,
    data: pd.Series,
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
        data_arr = data.to_numpy(dtype=np.float64)
    else:
        data_arr = data
    
    # Warm-up run
    _ = func(data_arr, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(5):
        start_time = time.perf_counter()
        result = func(data_arr, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result_shape': result.shape if hasattr(result, 'shape') else len(result)
    }

def run_comparison(num_symbols: int = 1000):
    """Run performance comparison between Python and Rust implementations.
    
    Args:
        num_symbols: Number of symbols in the test dataset
    """
    print(f"Creating dataset with {num_symbols} symbols...")
    data = create_large_dataset(num_symbols=num_symbols)
    prices = data['close']
    print(f"Dataset shape: {prices.shape}")
    
    # Define test cases
    test_cases = [
        {
            'name': 'Momentum Factor (252-day)',
            'implementations': {
                'python': (py_momentum, {'lookback': 252}),
                'rust': (rs_momentum, {'lookback': 252})
            }
        },
        {
            'name': 'Mean Reversion (20-day)',
            'implementations': {
                'python': (py_mean_reversion, {'lookback': 20}),
                'rust': (rs_mean_reversion, {'lookback': 20})
            }
        },
        {
            'name': 'Relative Strength (60-day)',
            'implementations': {
                'python': (py_relative_strength, {'lookback': 60}),
                'rust': (rs_relative_strength, {'lookback': 60})
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
            result = benchmark_implementation(func, prices, impl_name, **kwargs)
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
    # Run comparison with increasing dataset sizes
    for num_symbols in [1000, 2000, 5000]:
        print(f"\n{'='*80}")
        print(f"Running comparison with {num_symbols} symbols")
        print(f"{'='*80}")
        results = run_comparison(num_symbols) 