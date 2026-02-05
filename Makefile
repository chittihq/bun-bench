.PHONY: help install install-dev install-all test lint format typecheck build clean publish-test publish

PYTHON := python3
PIP := $(PYTHON) -m pip

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	$(PIP) install -e .

install-dev:  ## Install package with dev dependencies
	$(PIP) install -e ".[dev]"

install-all:  ## Install package with all dependencies
	$(PIP) install -e ".[all]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=bunbench --cov-report=html --cov-report=term

lint:  ## Run linting
	ruff check bunbench/
	black --check bunbench/

format:  ## Format code
	black bunbench/ tests/
	ruff check --fix bunbench/

typecheck:  ## Run type checking
	mypy bunbench/ --ignore-missing-imports

build:  ## Build package
	rm -rf dist/ build/ *.egg-info
	$(PYTHON) -m build

check:  ## Check built package
	twine check dist/*

clean:  ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

publish-test:  ## Publish to TestPyPI
	$(PYTHON) -m build
	twine check dist/*
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	$(PYTHON) -m build
	twine check dist/*
	twine upload dist/*

docker-base:  ## Build base Docker image
	docker build -f docker/Dockerfile.base -t bunbench.base .

docker-env:  ## Build env Docker image (BUN_VERSION=1.0.0)
	docker build -f docker/Dockerfile.env -t bunbench.env.$(BUN_VERSION) --build-arg BUN_VERSION=$(BUN_VERSION) .

validate:  ## Validate dataset
	$(PYTHON) -m bunbench validate dataset/tasks.json

evaluate:  ## Run evaluation (PREDICTIONS=path/to/predictions.json)
	$(PYTHON) -m bunbench evaluate -d dataset/tasks.json -p $(PREDICTIONS) -o results/
