.PHONY: help setup install install-dev lint format test clean docker-build docker-run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up development environment
	python -m pip install --upgrade pip uv
	uv pip install -e ".[dev]"
	pre-commit install

install: ## Install package in production mode
	python -m pip install --upgrade pip uv
	uv pip install .

install-dev: ## Install package in development mode
	python -m pip install --upgrade pip uv
	uv pip install -e ".[dev]"

lint: ## Run linting checks
	ruff check privacy_redactor_rt tests
	black --check privacy_redactor_rt tests
	isort --check-only privacy_redactor_rt tests
	mypy privacy_redactor_rt

format: ## Format code
	black privacy_redactor_rt tests
	isort privacy_redactor_rt tests
	ruff check --fix privacy_redactor_rt tests

test: ## Run tests
	pytest tests/ -v --cov=privacy_redactor_rt --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -v -x

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t privacy-redactor-rt .

docker-run: ## Run Docker container
	docker run -p 8501:8501 privacy-redactor-rt

run-app: ## Run Streamlit app locally
	streamlit run privacy_redactor_rt/app.py

run-cli: ## Show CLI help
	python -m privacy_redactor_rt.cli --help