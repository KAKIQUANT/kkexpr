"""
Expression evaluation module.

This module provides functions for evaluating string expressions
on financial data.
"""

import pandas as pd

from .expr_functions import *

def expr_transform(df: pd.DataFrame, expr: str) -> str:
    """Transform column references in expression.
    
    Args:
        df: Input DataFrame
        expr: Expression to transform
        
    Returns:
        Transformed expression with column references
    """
    # Replace column names with DataFrame references
    for col in df.columns:
        if col == 'pe' or col == 'pb':
            continue
        expr = expr.replace(col, 'df["{}"]'.format(col))
    return expr

def calc_expr(df: pd.DataFrame, expr: str) -> pd.Series:
    """Calculate expression on DataFrame.
    
    Args:
        df: Input DataFrame
        expr: Expression to evaluate
        
    Returns:
        Series with expression results
    """
    # If expression is a column name, return that column
    if expr in df.columns:
        return df[expr]
    
    # Transform and evaluate expression
    transformed_expr = expr_transform(df, expr)
    try:
        result = eval(transformed_expr)
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")
