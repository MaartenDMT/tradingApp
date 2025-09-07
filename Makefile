SHELL := /usr/bin/env pwsh

.PHONY: run test sync

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
