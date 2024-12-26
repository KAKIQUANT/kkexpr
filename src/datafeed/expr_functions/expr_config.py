from dataclasses import dataclass
from typing import Dict

@dataclass
class ExprConfig:
    """Configuration for expression calculation parameters."""
    
    # Default periods for different operations
    default_periods: Dict[str, int] = None
    
    # Minimum allowed periods for different operations
    min_periods: Dict[str, int] = None
    
    def __post_init__(self):
        if self.default_periods is None:
            self.default_periods = {
                'delay': 5,
                'delta': 20,
                'mean': 10,
                'median': 5,
                'max': 5,
                'min': 5,
                'std': 5,
                'rank': 9,
                'skew': 10,
                'kurt': 10,
                'argmax': 5,
                'argmin': 5,
            }
            
        if self.min_periods is None:
            self.min_periods = {
                'delay': 1,
                'delta': 1,
                'mean': 2,
                'median': 2,
                'max': 1,
                'min': 1,
                'std': 2,
                'rank': 2,
                'skew': 3,
                'kurt': 3,
                'argmax': 1,
                'argmin': 1,
            }

# Global configuration instance
config = ExprConfig() 