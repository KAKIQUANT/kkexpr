import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

from datafeed.expr_functions import (
    ts_delay, ts_delta, ts_mean, ts_median, ts_pct_change,
    ts_max, ts_min, ts_maxmin, ts_sum, ts_std, ts_skew,
    ts_kurt, ts_argmin, ts_argmax, ts_argmaxmin, ts_rank
)
from datafeed.expr_functions.expr_exceptions import InvalidPeriodError

def fix_series_name(result: pd.Series, expected_name: str) -> pd.Series:
    """Helper function to fix series name for comparison."""
    result.name = expected_name
    return result

def test_ts_delay(sample_series):
    """Test time series delay function."""
    result = ts_delay(sample_series, periods=1)
    expected = sample_series.groupby(level=1).shift(1)
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_delay_invalid_period(sample_series):
    """Test time series delay with invalid period."""
    with pytest.raises(InvalidPeriodError):
        ts_delay(sample_series, periods=0)

def test_ts_delta(sample_series):
    """Test time series delta function."""
    result = ts_delta(sample_series, periods=1)
    expected = sample_series - sample_series.groupby(level=1).shift(1)
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_mean(sample_series):
    """Test time series mean function."""
    result = ts_mean(sample_series, periods=3)
    expected = sample_series.groupby(level=1).rolling(window=3).mean()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_median(sample_series):
    """Test time series median function."""
    result = ts_median(sample_series, periods=3)
    expected = sample_series.groupby(level=1).rolling(window=3).median()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_pct_change(sample_series):
    """Test time series percentage change function."""
    result = ts_pct_change(sample_series, periods=1)
    expected = sample_series.groupby(level=1).pct_change(1)
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_max(sample_series):
    """Test time series maximum function."""
    result = ts_max(sample_series, periods=3)
    expected = sample_series.groupby(level=1).rolling(window=3).max()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_min(sample_series):
    """Test time series minimum function."""
    result = ts_min(sample_series, periods=3)
    expected = sample_series.groupby(level=1).rolling(window=3).min()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_maxmin(sample_series):
    """Test time series max-min normalization function."""
    result = ts_maxmin(sample_series, periods=3)
    min_val = ts_min(sample_series, 3)
    max_val = ts_max(sample_series, 3)
    expected = (sample_series - min_val) / (max_val - min_val)
    expected.name = result.name  # Use result's name for comparison
    assert_series_equal(result, expected)

def test_ts_sum(sample_series):
    """Test time series sum function."""
    result = ts_sum(sample_series, periods=3)
    expected = sample_series.groupby(level=1).rolling(window=3).sum()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_std(sample_series):
    """Test time series standard deviation function."""
    result = ts_std(sample_series, periods=3)
    expected = sample_series.groupby(level=1).rolling(window=3).std()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_skew(sample_series):
    """Test time series skewness function."""
    result = ts_skew(sample_series, periods=3)
    expected = sample_series.groupby(level=1).rolling(window=3).skew()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_kurt(sample_series):
    """Test time series kurtosis function."""
    result = ts_kurt(sample_series, periods=4)
    expected = sample_series.groupby(level=1).rolling(window=4).kurt()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_ts_rank(sample_series):
    """Test time series rank function."""
    result = ts_rank(sample_series, periods=3)
    def rank_pct(x):
        return pd.Series(x).rank(pct=True).iloc[-1]
    expected = sample_series.groupby(level=1).rolling(window=3).apply(rank_pct)
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_multi_symbol_calculation(multi_symbol_series):
    """Test calculation with multiple symbols."""
    result = ts_mean(multi_symbol_series, periods=3)
    expected = multi_symbol_series.groupby(level=1).rolling(window=3).mean()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    
    # Sort both result and expected by index for comparison
    result = result.sort_index()
    expected = expected.sort_index()
    
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_nan_handling(sample_series_with_nan):
    """Test handling of NaN values."""
    result = ts_mean(sample_series_with_nan, periods=3)
    expected = sample_series_with_nan.groupby(level=1).rolling(window=3).mean()
    expected.index = expected.index.droplevel(0)  # Remove extra level
    assert_series_equal(fix_series_name(result, expected.name), expected)

def test_default_periods():
    """Test default period values from config."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    index = pd.MultiIndex.from_product([dates, ['TEST']], names=['date', 'symbol'])
    series = pd.Series(range(10), index=index, name='test_series')
    
    # Test with default periods
    result = ts_delay(series)  # Should use default period from config
    assert not result.empty
    assert isinstance(result, pd.Series) 