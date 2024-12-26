import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from datafeed.metrics import (
    max_drawdown, sharpe_ratio, annual_return,
    sortino_ratio, calmar_ratio
)

@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01])

@pytest.fixture
def zero_returns():
    """Create zero returns for testing edge cases."""
    return pd.Series([0.0] * 10)

@pytest.fixture
def positive_returns():
    """Create positive returns for testing edge cases."""
    return pd.Series([0.01] * 10)

def test_max_drawdown(sample_returns):
    """Test maximum drawdown calculation."""
    result = max_drawdown(sample_returns)
    assert result > 0
    assert result <= 1

def test_max_drawdown_zero_returns(zero_returns):
    """Test maximum drawdown with zero returns."""
    result = max_drawdown(zero_returns)
    assert_almost_equal(result, 0)

def test_sharpe_ratio(sample_returns):
    """Test Sharpe ratio calculation."""
    result = sharpe_ratio(sample_returns, risk_free=0.0, periods=252)
    assert isinstance(result, float)
    assert not np.isnan(result)

def test_sharpe_ratio_zero_std(zero_returns):
    """Test Sharpe ratio with zero standard deviation."""
    result = sharpe_ratio(zero_returns)
    assert np.isnan(result)

def test_annual_return(sample_returns):
    """Test annualized return calculation."""
    result = annual_return(sample_returns, periods=252)
    assert isinstance(result, float)
    assert not np.isnan(result)

def test_annual_return_positive(positive_returns):
    """Test annualized return with all positive returns."""
    result = annual_return(positive_returns, periods=252)
    assert result > 0

def test_sortino_ratio(sample_returns):
    """Test Sortino ratio calculation."""
    result = sortino_ratio(sample_returns, risk_free=0.0, periods=252)
    assert isinstance(result, float)
    assert not np.isnan(result)

def test_sortino_ratio_positive_returns(positive_returns):
    """Test Sortino ratio with all positive returns."""
    result = sortino_ratio(positive_returns)
    assert result == np.inf

def test_calmar_ratio(sample_returns):
    """Test Calmar ratio calculation."""
    result = calmar_ratio(sample_returns, periods=252)
    assert isinstance(result, float)
    assert not np.isnan(result)

def test_calmar_ratio_zero_drawdown(positive_returns):
    """Test Calmar ratio with zero drawdown."""
    result = calmar_ratio(positive_returns)
    assert np.isnan(result)

def test_empty_input():
    """Test all metrics with empty input."""
    empty = pd.Series([], dtype=float)  # Explicitly set dtype to float
    assert np.isnan(max_drawdown(empty))
    assert np.isnan(sharpe_ratio(empty))
    assert np.isnan(annual_return(empty))
    assert np.isnan(sortino_ratio(empty))
    assert np.isnan(calmar_ratio(empty))

def test_numpy_array_input():
    """Test all metrics with numpy array input."""
    arr = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    assert isinstance(max_drawdown(arr), float)
    assert isinstance(sharpe_ratio(arr), float)
    assert isinstance(annual_return(arr), float)
    assert isinstance(sortino_ratio(arr), float)
    assert isinstance(calmar_ratio(arr), float) 