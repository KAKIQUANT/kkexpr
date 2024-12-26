"""
High-level API for factor computation.

This module provides a simple interface for computing factors on financial data.
"""

from typing import Union, Dict, Any, Optional
import pandas as pd
import numpy as np

from .expr import calc_expr
from .rust_expr import (
    momentum_factor as rs_momentum,
    mean_reversion_factor as rs_mean_reversion,
    alpha101_factor_42 as rs_alpha42
)

class FactorEngine:
    """Engine for computing factors on financial data."""
    
    def __init__(self, use_rust: bool = False):
        """Initialize the factor engine.
        
        Args:
            use_rust: Whether to use Rust implementations when available
        """
        self.use_rust = use_rust
        self.factor_expressions = {
            'momentum': self._momentum_expr,
            'mean_reversion': self._mean_reversion_expr,
            'relative_strength': self._relative_strength_expr,
            'alpha42': self._alpha42_expr
        }
    
    def execute_factor(
        self,
        data: pd.DataFrame,
        factor_name: str,
        **kwargs
    ) -> pd.Series:
        """Execute a factor computation on the input data.
        
        Args:
            data: Input DataFrame with financial data
            factor_name: Name of the factor to compute
            **kwargs: Additional arguments for the factor
            
        Returns:
            Series with factor values
        """
        if factor_name not in self.factor_expressions:
            raise ValueError(f"Unknown factor: {factor_name}")
        
        if self.use_rust:
            # Use Rust implementation if available
            if factor_name == 'momentum':
                result = rs_momentum(data['close'].to_numpy(dtype=np.float64), kwargs.get('lookback', 252))
            elif factor_name == 'mean_reversion':
                result = rs_mean_reversion(data['close'].to_numpy(dtype=np.float64), kwargs.get('lookback', 20))
            elif factor_name == 'alpha42':
                result = rs_alpha42(data['high'].to_numpy(dtype=np.float64), data['volume'].to_numpy(dtype=np.float64))
            else:
                # Fall back to Python implementation
                expr = self.factor_expressions[factor_name](**kwargs)
                result = calc_expr(data, expr)
        else:
            # Use Python implementation
            expr = self.factor_expressions[factor_name](**kwargs)
            result = calc_expr(data, expr)
        
        return pd.Series(result, index=data.index, name=factor_name)
    
    def _momentum_expr(self, lookback: int = 252) -> str:
        """Get momentum factor expression."""
        return f"ts_pct_change(close, {lookback}) / ts_std(ts_pct_change(close, 1), {lookback})"
    
    def _mean_reversion_expr(self, lookback: int = 20) -> str:
        """Get mean reversion factor expression."""
        return f"-(close - ts_mean(close, {lookback})) / ts_std(close, {lookback})"
    
    def _relative_strength_expr(self, lookback: int = 60) -> str:
        """Get relative strength factor expression."""
        tf1 = lookback // 3
        tf2 = lookback
        tf3 = lookback * 2
        return (
            f"0.5 * ts_rank(ts_pct_change(close, {tf1}), {lookback}) + "
            f"0.3 * ts_rank(ts_pct_change(close, {tf2}), {lookback}) + "
            f"0.2 * ts_rank(ts_pct_change(close, {tf3}), {lookback})"
        )
    
    def _alpha42_expr(self) -> str:
        """Get Alpha101 Factor #42 expression."""
        return "-ts_rank(ts_std(high, 10), 10) * ts_correlation(high, volume, 10)" 