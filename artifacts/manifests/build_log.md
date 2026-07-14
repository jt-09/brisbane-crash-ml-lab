# Build log — Brisbane Crash ML Lab

## Phase A — Bootstrap foundation (2026-07-14)

- Replaced CLI stub with Typer command surface (`version`, `doctor`, staged pipeline hooks).
- Added `config.py` (YAML profiles with `inherits`, path resolution), `logging.py`, `paths.py`, and `data/manifest.py`.
- `make smoke` runs unit tests, `crashlab version`, `doctor`, and bootstrap `all --profile smoke`.
- Full acquisition/ML stages remain Phase B+.

## Phase A — Implementation complete (2026-07-14)

- Bootstrap foundation implemented: config inheritance, logging, paths, manifest helpers, Typer CLI, Makefile targets.
- Unit tests: **8 passed** (`tests/unit`: config, paths, version).
- Package version: **0.1.0a1**.
- `crashlab all --profile smoke` completes bootstrap planning only (`status: bootstrap_plan_only`); acquisition and ML stages deferred to Phase B+.
