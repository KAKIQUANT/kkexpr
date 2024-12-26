"""
Simple demonstration of factor computation.

This example shows how to:
1. Load data for two SH indexes
2. Compute three WorldQuant factors
3. Compare Python vs Rust implementations
"""

import pandas as pd
import os
import glob
import numpy as np
import time
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

from datafeed.factor_engine import FactorEngine

def load_two_indexes(data_dir: str = "data/indexes", symbols: list = ['000300.SH', '000905.SH']) -> pd.DataFrame:
    """Load data for two specific SH indexes.
    
    Args:
        data_dir: Directory containing index data files
        symbols: List of index symbols to load (default: CSI 300 and CSI 500)
        
    Returns:
        DataFrame with the specified indexes' data
    """
    dfs = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found for {symbol}")
        
        # Read CSV and add symbol column
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        dfs.append(df)
    
    # Stack all dataframes
    data = pd.concat(dfs, axis=0)
    
    # Set multi-index
    data = data.set_index(['date', 'symbol']).sort_index()
    
    # Drop any duplicate columns (like 'symbol' that might be in the CSV)
    data = data[['open', 'high', 'close', 'low', 'volume']]
    
    return data

def compute_factors(data: pd.DataFrame, enable_rust: bool = False) -> pd.DataFrame:
    """Compute factors using either Python or Rust implementation.
    
    Args:
        data: Input DataFrame
        enable_rust: Whether to use Rust implementation
        
    Returns:
        DataFrame with factor values
    """
    # Initialize factor engine
    engine = FactorEngine(use_rust=enable_rust)
    
    # Define factors to compute
    factors = {
        'momentum': {'lookback': 20},  # Short-term momentum
        'mean_reversion': {'lookback': 10},  # Short-term mean reversion
        'alpha42': {}  # Volume-price correlation factor
    }
    
    # Compute each factor
    results = pd.DataFrame(index=data.index)
    impl = "Rust" if enable_rust else "Python"
    print(f"\nComputing factors using {impl} implementation...")
    
    for factor_name, kwargs in factors.items():
        print(f"\nComputing {factor_name}...")
        if not enable_rust:
            expr = engine.factor_expressions[factor_name](**kwargs)
            print(f"Expression: {expr}")
        
        start_time = time.time()
        results[factor_name] = engine.execute_factor(data, factor_name, **kwargs)
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.4f} seconds")
    
    return results

def compare_implementations(data: pd.DataFrame):
    """Compare Python and Rust implementations.
    
    Args:
        data: Input DataFrame
    """
    # Compute factors using both implementations
    py_results = compute_factors(data, enable_rust=False)
    rust_results = compute_factors(data, enable_rust=True)
    
    # Compare results
    print("\nComparison of Python vs Rust implementations:")
    for factor_name in py_results.columns:
        print(f"\n{factor_name}:")
        py_vals = py_results[factor_name]
        rust_vals = rust_results[factor_name]
        
        # Calculate differences
        diff = py_vals - rust_vals
        valid_diff = diff[~(np.isnan(diff) | np.isinf(diff))]
        
        print(f"Max absolute difference: {np.abs(valid_diff).max():.2e}")
        print(f"Mean absolute difference: {np.abs(valid_diff).mean():.2e}")
        print(f"Correlation: {py_vals.corr(rust_vals):.4f}")
    
    # Show combined statistics
    all_results = pd.concat([
        py_results.add_suffix('_py'),
        rust_results.add_suffix('_rust')
    ], axis=1)
    print("\nCombined Factor Statistics:")
    print(all_results.describe().round(4))

def main():
    """Run the simple factor computation demonstration."""
    print("Loading index data...")
    data = load_two_indexes()
    print(f"Dataset shape: {data.shape}")
    print("\nFirst few rows of input data:")
    print(data.head())
    
    # Compare Python and Rust implementations
    compare_implementations(data)

if __name__ == '__main__':
    main() 