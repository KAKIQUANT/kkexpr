import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_series():
    """Create a sample time series for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    values = [1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 4.0, 3.0, 2.0, 1.0]
    index = pd.MultiIndex.from_product([dates, ['TEST']], names=['date', 'symbol'])
    return pd.Series(values, index=index, name='test_series')

@pytest.fixture
def sample_series_with_nan():
    """Create a sample time series with NaN values for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    values = [1.0, np.nan, 3.0, 2.0, np.nan, 3.0, 4.0, np.nan, 2.0, 1.0]
    index = pd.MultiIndex.from_product([dates, ['TEST']], names=['date', 'symbol'])
    return pd.Series(values, index=index, name='test_series')

@pytest.fixture
def multi_symbol_series():
    """Create a sample time series with multiple symbols for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
    symbols = ['A', 'B']
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    values = [1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 3.0, 2.0, 2.0, 3.0]
    return pd.Series(values, index=index, name='test_series') 