"""
Rust implementation of expression functions.
"""

import os
import sys

# Add the current directory to PATH for DLL loading
os.environ["PATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ.get("PATH", "")

from .rust_expr import momentum_factor, mean_reversion_factor, relative_strength_factor, alpha101_factor_42

__all__ = [
    'momentum_factor',
    'mean_reversion_factor',
    'relative_strength_factor',
    'alpha101_factor_42'
] 