# Tasman Agentic Analytics - Makefile
# Simple commands for common tasks

.PHONY: help install test clean run sample-data lint format check

# Default target
help:
	@echo "Tasman Agentic Analytics - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "  make install      - Install dependencies and setup environment"
	@echo "  make sample-data  - Generate sample DuckDB database"
	@echo "  make test         - Run test suite"
	@echo "  make test-v       - Run tests with verbose output"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make run          - Launch Jupyter notebook"
	@echo "  make lint         - Check code style (if ruff installed)"
	@echo "  make format       - Format code (if ruff installed)"
	@echo "  make clean        - Remove cache and temporary files"
	@echo "  make check        - Run all checks (test + lint)"
	@echo ""

# Install dependencies and setup environment
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync
	@echo "✅ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and add your API keys (optional)"
	@echo "  2. Run 'make sample-data' to create sample database"
	@echo "  3. Run 'make test' to verify installation"
	@echo "  4. Run 'make run' to launch Jupyter notebook"

# Generate sample database
sample-data:
	@echo "🦆 Generating sample DuckDB database..."
	uv run python scripts/create_sample_data.py
	@echo "✅ Sample data created at ./data/marketing.duckdb"

# Run test suite
test:
	@echo "🧪 Running test suite..."
	uv run pytest tests/ -v

# Run tests with verbose output
test-v:
	@echo "🧪 Running test suite (verbose)..."
	uv run pytest tests/ -vv --tb=short

# Run tests with coverage
test-cov:
	@echo "🧪 Running test suite with coverage..."
	uv run pytest tests/ --cov=core --cov=agents --cov-report=term-missing

# Launch Jupyter notebook
run:
	@echo "🚀 Launching Jupyter notebook..."
	@echo "The notebook will open in your browser at http://localhost:8888"
	@echo ""
	uv run jupyter notebook notebooks/Agentic_Analytics_Demo.ipynb

# Lint code (if ruff is available)
lint:
	@echo "🔍 Checking code style..."
	@if command -v ruff >/dev/null 2>&1; then \
		uv run ruff check core agents tests scripts; \
	else \
		echo "ℹ️  Ruff not installed. Run 'uv add --dev ruff' to enable linting."; \
	fi

# Format code (if ruff is available)
format:
	@echo "✨ Formatting code..."
	@if command -v ruff >/dev/null 2>&1; then \
		uv run ruff format core agents tests scripts; \
		uv run ruff check --fix core agents tests scripts; \
	else \
		echo "ℹ️  Ruff not installed. Run 'uv add --dev ruff' to enable formatting."; \
	fi

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .cache/llm/*.json 2>/dev/null || true
	rm -rf notebooks/.ipynb_checkpoints 2>/dev/null || true
	rm -rf notebooks/outputs/*.png 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Run all checks
check: test lint
	@echo "✅ All checks passed!"
