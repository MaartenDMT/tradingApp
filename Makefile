SHELL := /usr/bin/env pwsh

.PHONY: run test sync install-deps clean help dev-setup lint format \
        build-docker run-docker push-docker docs security-check \
        performance-test coverage

# Default target
help:
	@echo "Trading Application Makefile"
	@echo "============================"
	@echo "Available targets:"
	@echo "  run              - Run the main application"
	@echo "  test             - Run tests"
	@echo "  sync             - Sync project dependencies"
	@echo "  install-deps     - Install dependencies via install_deps.py"
	@echo "  dev-setup        - Set up development environment"
	@echo "  clean            - Clean build artifacts"
	@echo "  lint             - Run code linting"
	@echo "  format           - Format code with black"
	@echo "  docs             - Generate documentation"
	@echo "  security-check   - Run security checks"
	@echo "  performance-test - Run performance tests"
	@echo "  coverage         - Generate test coverage report"
	@echo "  build-docker     - Build Docker image"
	@echo "  run-docker       - Run application in Docker"
	@echo "  push-docker      - Push Docker image to registry"
	@echo "  help             - Show this help message"

# Run the main application using uv
run:
	@echo "Running main.py via uv..."
	uv run python main.py

# Run tests using uv
test:
	@echo "Running pytest via uv..."
	uv run pytest

# Sync project dependencies using uv
sync:
	@echo "Synchronizing dependencies via uv..."
	uv sync

# Install dependencies using our custom script
install-deps:
	@echo "Installing dependencies via install_deps.py..."
	python install_deps.py

# Set up development environment
dev-setup: sync install-deps
	@echo "Development environment set up successfully!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean completed!"

# Run code linting
lint:
	@echo "Running code linting..."
	uv run flake8 .
	uv run pylint model/ view/ util/ *.py

# Format code with black
format:
	@echo "Formatting code with black..."
	uv run black .

# Generate documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here
	@echo "Documentation generation completed!"

# Run security checks
security-check:
	@echo "Running security checks..."
	uv run bandit -r .
	uv run safety check

# Run performance tests
performance-test:
	@echo "Running performance tests..."
	uv run python test/test_performance.py

# Generate test coverage report
coverage:
	@echo "Generating test coverage report..."
	uv run pytest --cov=. --cov-report=html --cov-report=term

# Docker targets
build-docker:
	@echo "Building Docker image..."
	docker build -t trading-app:latest .

run-docker:
	@echo "Running application in Docker..."
	docker run -it --rm -p 8080:8080 trading-app:latest

push-docker:
	@echo "Pushing Docker image to registry..."
	docker tag trading-app:latest your-registry/trading-app:latest
	docker push your-registry/trading-app:latest
