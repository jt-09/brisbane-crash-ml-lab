.PHONY: setup lint typecheck test smoke help

help:
	@echo "Pre-build targets only. Full Makefile is implemented by the autonomous build agent."
	@echo "  make setup   - uv sync"
	@echo "  make lint    - ruff check + format check"
	@echo "  make typecheck - mypy src"
	@echo "  make test    - pytest"

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	uv run pytest

smoke:
	@echo "Smoke pipeline not implemented yet — autonomous agent owns Phase A+."
	uv run crashlab version
