"""
Demonstration of the factor engine API.

This module shows how to use the high-level factor computation API
with real market data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from src.datafeed.factor_engine import FactorEngine

def load_index_data(data_dir: str = "data/indexes") -> pd.DataFrame:
    """Load and stack all SH index data.
    
    Args:
        data_dir: Directory containing index data files
        
    Returns:
        DataFrame with stacked index data
    """
    import os
    import glob
    
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

def compute_factors(data: pd.DataFrame, use_rust: bool = True) -> Dict[str, pd.Series]:
    """Compute multiple factors using the factor engine.
    
    Args:
        data: Input DataFrame with financial data
        use_rust: Whether to use Rust implementations
        
    Returns:
        Dictionary mapping factor names to their values
    """
    engine = FactorEngine(use_rust=use_rust)
    
    # Define factors to compute
    factors = {
        'momentum': {'lookback': 252},
        'mean_reversion': {'lookback': 20},
        'relative_strength': {'lookback': 60},
        'alpha42': {}
    }
    
    # Compute each factor
    results = {}
    for factor_name, kwargs in factors.items():
        print(f"\nComputing {factor_name}...")
        results[factor_name] = engine.execute_factor(data, factor_name, **kwargs)
    
    return results

def analyze_factors(factors: Dict[str, pd.Series]) -> None:
    """Analyze and plot factor distributions and correlations.
    
    Args:
        factors: Dictionary mapping factor names to their values
    """
    # Convert factors to DataFrame
    factor_df = pd.DataFrame(factors)
    
    # Plot factor distributions
    plt.figure(figsize=(12, 6))
    for i, (name, values) in enumerate(factors.items(), 1):
        plt.subplot(2, 2, i)
        # Filter out NaN and infinite values
        valid_values = values[~(np.isnan(values) | np.isinf(values))]
        if len(valid_values) > 0:
            valid_values.hist(bins=50)
            plt.title(f"{name} Distribution")
            plt.xlabel("Factor Value")
            plt.ylabel("Frequency")
        else:
            plt.text(0.5, 0.5, "No valid values", ha='center', va='center')
            plt.title(f"{name} (No Data)")
    plt.tight_layout()
    plt.show()
    
    # Calculate and plot correlation matrix
    plt.figure(figsize=(8, 6))
    corr = factor_df.corr()
    plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Factor Correlations")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center')
    plt.tight_layout()
    plt.show()
    
    # Print factor statistics
    print("\nFactor Statistics:")
    print(factor_df.describe())

def main():
    """Run the factor engine demonstration."""
    print("Loading index data...")
    data = load_index_data()
    print(f"Dataset shape: {data.shape}")
    
    # Compute factors using both Python and Rust
    print("\nComputing factors using Python...")
    py_factors = compute_factors(data, use_rust=False)
    
    print("\nComputing factors using Rust...")
    rs_factors = compute_factors(data, use_rust=True)
    
    # Analyze Python implementation results
    print("\nAnalyzing Python implementation results:")
    analyze_factors(py_factors)
    
    # Analyze Rust implementation results
    print("\nAnalyzing Rust implementation results:")
    analyze_factors(rs_factors)
    
    # Compare implementations
    print("\nComparing implementations:")
    for factor_name in py_factors:
        py_vals = py_factors[factor_name]
        rs_vals = rs_factors[factor_name]
        
        # Calculate differences
        diff = py_vals - rs_vals
        max_diff = np.abs(diff).max()
        mean_diff = np.abs(diff).mean()
        
        print(f"\n{factor_name}:")
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")

if __name__ == '__main__':
    main() 