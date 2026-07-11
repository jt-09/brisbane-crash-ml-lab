# Brisbane Crash ML Lab

CPU-first, reproducible ML lab on Brisbane City road-crash records (Queensland open data).

## Quick start

```bash
uv sync
uv run crashlab version
uv run crashlab doctor --profile smoke
```

### Smoke pipeline (fixture, no network, &lt;5 min)

```bash
uv run crashlab all --profile smoke --force
```

### Standard pipeline (local Brisbane extract)

```bash
uv run crashlab acquire --profile standard
uv run crashlab all --profile standard
```

### Individual stages

```bash
uv run crashlab validate --profile standard
uv run crashlab prepare --profile standard
uv run crashlab train-binary --profile standard
uv run crashlab train-multiclass --profile standard
uv run crashlab train-ordinal --profile standard
uv run crashlab detect-anomalies --profile standard
uv run crashlab cluster-hotspots --profile standard
uv run crashlab train-counts --profile standard
uv run crashlab report --profile standard
```

### Reports and app

```bash
uv run crashlab report --profile smoke
# → reports/index.html

make app
# → uv run streamlit run src/crashlab/app/streamlit_app.py
```

### Quality checks

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
uv run mypy src
uv run pytest -q
make smoke
```

## Outputs

| Output | Location |
|--------|----------|
| Cleaned data | `data/processed/brisbane_crashes_cleaned.parquet` |
| Model metrics | `artifacts/metrics/` |
| HTML report | `reports/index.html` |
| Figures / tables | `reports/figures/`, `reports/tables/` |

## Interpretation

Models describe **predictive associations** on reported crashes. They do not establish causation or operational road-safety guidance. See `docs/model_card.md` and `docs/data_provenance.md`.

## Specs

- [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) — binding specification
- [`docs/architecture.md`](docs/architecture.md) — layout and artifacts

## License / attribution

Code: MIT (see `LICENSE`).
Crash data: © State of Queensland — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
