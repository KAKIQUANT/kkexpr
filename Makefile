.PHONY: install test coverage clean format lint

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

all: clean install format lint test coverage 