# KKExpr - Factor Expression Engine

A high-performance factor expression execution engine for A-share market data analysis.

## Features

- Fast and efficient factor calculation
- Support for multiple data sources
- Comprehensive time series operations
- Multi-symbol processing
- Extensible expression system

## Installation

```bash
# Development installation
make install

# Or using pip directly
pip install -e ".[dev]"
```

## Usage

```python
from datafeed.dataloader import CSVDataLoader
from datafeed.expr_functions import ts_mean, ts_std

# Initialize data loader
loader = CSVDataLoader()

# Load data
df = loader.get_df(symbols=['000001.SZ', '000002.SZ'], start_date='20220101')

# Calculate expressions
df = loader.calc_expr(df, 
    fields=['ts_mean(close,20)', 'ts_std(close,20)'],
    names=['ma20', 'std20']
)
```

## Development

### Setup

```bash
# Install development dependencies
make install
```

### Testing

```bash
# Run tests
make test

# Run tests with coverage report
make coverage
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint
```

### Clean

```bash
# Clean build artifacts
make clean
```

## Project Structure

```
kkexpr/
├── src/
│   ├── data/           # Data handling
│   └── datafeed/       # Data loading and processing
│       └── expr_functions/  # Expression implementations
├── tests/              # Test cases
├── setup.py           # Package configuration
└── Makefile          # Development commands
```

## License

MIT License 