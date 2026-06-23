.PHONY: setup lint typecheck test smoke download validate prepare train anomalies spatial counts report all app clean-generated help

PROFILE ?= standard
SMOKE_PROFILE ?= smoke

help:
	@echo "Brisbane Crash ML Lab — Make targets"
	@echo "  make setup          - uv sync"
	@echo "  make lint           - ruff check + format check"
	@echo "  make typecheck      - mypy src"
	@echo "  make test           - pytest"
	@echo "  make smoke          - bootstrap smoke (unit tests + version + doctor)"
	@echo "  make download       - crashlab acquire"
	@echo "  make validate       - crashlab validate"
	@echo "  make prepare        - crashlab prepare"
	@echo "  make train          - binary/multiclass/ordinal training"
	@echo "  make anomalies      - detect-anomalies"
	@echo "  make spatial        - cluster-hotspots"
	@echo "  make counts         - train-counts"
	@echo "  make report         - crashlab report"
	@echo "  make all            - crashlab all (standard profile)"
	@echo "  make app            - streamlit app"
	@echo "  make clean-generated - remove generated artifacts"

setup:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

smoke:
	uv run pytest tests/unit -q
	uv run crashlab version
	uv run crashlab doctor --profile $(SMOKE_PROFILE)
	uv run crashlab all --profile $(SMOKE_PROFILE)

download:
	uv run crashlab acquire --profile $(PROFILE)

validate:
	uv run crashlab validate --profile $(PROFILE)

prepare:
	uv run crashlab prepare --profile $(PROFILE)

train:
	uv run crashlab train-binary --profile $(PROFILE)
	uv run crashlab train-multiclass --profile $(PROFILE)
	uv run crashlab train-ordinal --profile $(PROFILE)

anomalies:
	uv run crashlab detect-anomalies --profile $(PROFILE)

spatial:
	uv run crashlab cluster-hotspots --profile $(PROFILE)

counts:
	uv run crashlab train-counts --profile $(PROFILE)

report:
	uv run crashlab report --profile $(PROFILE)

all:
	uv run crashlab all --profile $(PROFILE)

app:
	uv run streamlit run src/crashlab/app/streamlit_app.py

clean-generated:
	rm -rf artifacts reports data/interim data/processed
