use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use ndarray::{Array1, s};
use std::collections::VecDeque;

#[derive(thiserror::Error, Debug)]
pub enum ExprError {
    #[error("Invalid period: {0}")]
    InvalidPeriod(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
}

type Result<T> = std::result::Result<T, ExprError>;

/// Calculate rolling mean
fn rolling_mean(data: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
    if window < 1 {
        return Err(ExprError::InvalidPeriod("Window size must be positive".into()));
    }

    let n = data.len();
    let mut result = Array1::zeros(n);
    let mut sum = 0.0;
    let mut count = 0;
    let mut queue = VecDeque::with_capacity(window);

    for (i, &val) in data.iter().enumerate() {
        queue.push_back(val);
        sum += val;
        count += 1;

        if count > window {
            sum -= queue.pop_front().unwrap();
            count -= 1;
        }

        result[i] = if count < window {
            f64::NAN
        } else {
            sum / count as f64
        };
    }

    Ok(result)
}

/// Calculate rolling standard deviation
fn rolling_std(data: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
    if window < 2 {
        return Err(ExprError::InvalidPeriod("Window size must be at least 2".into()));
    }

    let n = data.len();
    let mut result = Array1::zeros(n);
    let mut queue = VecDeque::with_capacity(window);
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0;

    for (i, &val) in data.iter().enumerate() {
        if val.is_nan() || val.is_infinite() {
            result[i] = f64::NAN;
            continue;
        }

        queue.push_back(val);
        sum += val;
        sum_sq += val * val;
        count += 1;

        if count > window {
            let old = queue.pop_front().unwrap();
            sum -= old;
            sum_sq -= old * old;
            count -= 1;
        }

        result[i] = if count < window {
            f64::NAN
        } else {
            let mean = sum / count as f64;
            let variance = (sum_sq / count as f64) - (mean * mean);
            if variance <= 0.0 {
                f64::NAN
            } else {
                variance.sqrt()
            }
        };
    }

    Ok(result)
}

/// Calculate percentage change
fn pct_change(data: &Array1<f64>, periods: usize) -> Result<Array1<f64>> {
    if periods < 1 {
        return Err(ExprError::InvalidPeriod("Period must be positive".into()));
    }

    let n = data.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        result[i] = if i < periods {
            f64::NAN
        } else {
            let prev = data[i - periods];
            if prev.is_nan() || prev.is_infinite() || prev == 0.0 {
                f64::NAN
            } else {
                let curr = data[i];
                if curr.is_nan() || curr.is_infinite() {
                    f64::NAN
                } else {
                    (curr - prev) / prev
                }
            }
        };
    }

    Ok(result)
}

/// Calculate rolling rank (percentile)
fn rolling_rank(data: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
    if window < 2 {
        return Err(ExprError::InvalidPeriod("Window size must be at least 2".into()));
    }

    let n = data.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        result[i] = if i < window - 1 {
            f64::NAN
        } else {
            let start = i.saturating_sub(window - 1);
            let window_data: Vec<f64> = data.slice(s![start..=i])
                .iter()
                .filter(|&&x| !x.is_nan() && !x.is_infinite())
                .copied()
                .collect();
            
            if window_data.is_empty() {
                f64::NAN
            } else {
                let current = data[i];
                if current.is_nan() || current.is_infinite() {
                    f64::NAN
                } else {
                    let rank = window_data.iter()
                        .filter(|&&x| x <= current)
                        .count() as f64;
                    rank / window_data.len() as f64
                }
            }
        };
    }

    Ok(result)
}

/// Calculate rolling correlation
fn rolling_correlation(x: &Array1<f64>, y: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
    if window < 2 {
        return Err(ExprError::InvalidPeriod("Window size must be at least 2".into()));
    }

    let n = x.len();
    let mut result = Array1::zeros(n);
    let mut queue_x = VecDeque::with_capacity(window);
    let mut queue_y = VecDeque::with_capacity(window);
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut count = 0;

    for i in 0..n {
        if x[i].is_nan() || x[i].is_infinite() || y[i].is_nan() || y[i].is_infinite() {
            result[i] = f64::NAN;
            continue;
        }

        queue_x.push_back(x[i]);
        queue_y.push_back(y[i]);
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
        sum_yy += y[i] * y[i];
        count += 1;

        if count > window {
            let old_x = queue_x.pop_front().unwrap();
            let old_y = queue_y.pop_front().unwrap();
            sum_x -= old_x;
            sum_y -= old_y;
            sum_xy -= old_x * old_y;
            sum_xx -= old_x * old_x;
            sum_yy -= old_y * old_y;
            count -= 1;
        }

        result[i] = if count < window {
            f64::NAN
        } else {
            let mean_x = sum_x / count as f64;
            let mean_y = sum_y / count as f64;
            let cov = (sum_xy / count as f64) - (mean_x * mean_y);
            let var_x = (sum_xx / count as f64) - (mean_x * mean_x);
            let var_y = (sum_yy / count as f64) - (mean_y * mean_y);
            
            if var_x <= 0.0 || var_y <= 0.0 {
                f64::NAN
            } else {
                cov / (var_x.sqrt() * var_y.sqrt())
            }
        };
    }

    Ok(result)
}

/// Alpha101 Factor #42 calculation
#[pyfunction]
fn alpha101_factor_42<'py>(py: Python<'py>, high: &PyArray1<f64>, volume: &PyArray1<f64>) -> PyResult<&'py PyArray1<f64>> {
    let high = high.readonly();
    let volume = volume.readonly();
    let high_arr = Array1::from_vec(high.as_array().to_vec());
    let volume_arr = Array1::from_vec(volume.as_array().to_vec());
    
    // Calculate standard deviation of high prices
    let high_std = rolling_std(&high_arr, 10)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    // Calculate rank of standard deviation
    let vol_rank = rolling_rank(&high_std, 10)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    // Calculate correlation between high and volume
    let vol_price_corr = rolling_correlation(&high_arr, &volume_arr, 10)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    // Combine components
    let result = vol_rank.mapv(|x| if x.is_nan() || x.is_infinite() { f64::NAN } else { -x }) 
        * vol_price_corr.mapv(|x| if x.is_nan() || x.is_infinite() { f64::NAN } else { x });
    
    Ok(result.into_pyarray(py))
}

/// Python module
#[pymodule]
fn rust_expr(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Momentum factor calculation
    #[pyfunction]
    fn momentum_factor<'py>(py: Python<'py>, prices: &PyArray1<f64>, lookback: usize) -> PyResult<&'py PyArray1<f64>> {
        let prices = prices.readonly();
        let prices_arr = Array1::from_vec(prices.as_array().to_vec());
        
        // Calculate returns and momentum
        let returns = pct_change(&prices_arr, 1).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let momentum = pct_change(&prices_arr, lookback).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let vol = rolling_std(&returns, lookback).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Calculate momentum/vol ratio with NaN handling
        let result = momentum.iter()
            .zip(vol.iter())
            .map(|(&m, &v)| {
                if m.is_nan() || m.is_infinite() || v.is_nan() || v.is_infinite() || v == 0.0 {
                    f64::NAN
                } else {
                    m / v
                }
            })
            .collect::<Vec<f64>>();
        
        Ok(Array1::from_vec(result).into_pyarray(py))
    }

    /// Mean reversion factor calculation
    #[pyfunction]
    fn mean_reversion_factor<'py>(py: Python<'py>, prices: &PyArray1<f64>, lookback: usize) -> PyResult<&'py PyArray1<f64>> {
        let prices = prices.readonly();
        let prices_arr = Array1::from_vec(prices.as_array().to_vec());
        
        let ma = rolling_mean(&prices_arr, lookback).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let std = rolling_std(&prices_arr, lookback).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Calculate z-score with NaN handling
        let result = prices_arr.iter()
            .zip(ma.iter().zip(std.iter()))
            .map(|(&x, (&m, &s))| {
                if x.is_nan() || x.is_infinite() || m.is_nan() || m.is_infinite() || s.is_nan() || s.is_infinite() || s == 0.0 {
                    f64::NAN
                } else {
                    -(x - m) / s
                }
            })
            .collect::<Vec<f64>>();
        
        Ok(Array1::from_vec(result).into_pyarray(py))
    }

    /// Relative strength factor calculation
    #[pyfunction]
    fn relative_strength_factor<'py>(py: Python<'py>, prices: &PyArray1<f64>, lookback: usize) -> PyResult<&'py PyArray1<f64>> {
        let prices = prices.readonly();
        let prices_arr = Array1::from_vec(prices.as_array().to_vec());
        
        let timeframes = [lookback / 3, lookback, lookback * 2];
        let weights = [0.5, 0.3, 0.2];
        let mut result = Array1::zeros(prices_arr.len());
        
        for (&tf, &weight) in timeframes.iter().zip(weights.iter()) {
            let mom = pct_change(&prices_arr, tf).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let rank = rolling_rank(&mom, lookback).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            result = result + rank.mapv(|x| if x.is_nan() || x.is_infinite() { 0.0 } else { x * weight });
        }
        
        Ok(result.into_pyarray(py))
    }

    // Add functions to the module
    m.add_function(wrap_pyfunction!(momentum_factor, m)?)?;
    m.add_function(wrap_pyfunction!(mean_reversion_factor, m)?)?;
    m.add_function(wrap_pyfunction!(relative_strength_factor, m)?)?;

    // Add Alpha101 Factor #42
    m.add_function(wrap_pyfunction!(alpha101_factor_42, m)?)?;

    Ok(())
} 