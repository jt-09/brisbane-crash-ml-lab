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

## Phase B — Data acquisition (2026-07-14)

- Implemented `src/crashlab/data/schema.py` (canonical fields, ODS aliases, severity vocabulary, preseed SHA constant).
- Implemented `src/crashlab/data/acquire.py`: OpenDataSoft filtered export, preseed short-circuit, fixture copy for smoke, `CRASHLAB_ALLOW_NETWORK=0` gate, acquisition manifests.
- Preseed policy: reuse `data/raw/brisbane_crashes_2015_2023.csv` when SHA-256 matches `CE2A0435366E0870F238295CB9A2A700C12F6FFB25815B8D51ACA2A506CEFF22` (31,046 rows, ~9.45 MiB).
- Smoke profile writes `data/raw/fixture_smoke.csv` via `fixture_raw_path`; must not target `preseed_raw.path`.
- **Incident:** early smoke runs briefly clobbered the preseed CSV; file restored from OpenDataSoft with matching SHA; `_guard_fixture_destination` and dedicated `fixture_raw_path` added to prevent recurrence.
- CLI `acquire` command wired; `pipeline.run_all` runs acquire for implemented profiles.

## Phase C — Validate and clean (2026-07-14, same branch)

- Validation and cleaning delivered on `feat/data-acquisition` with acquisition for end-to-end smoke delivery (documented in `docs/prs/02-data-acquisition.md`).
- Implemented `src/crashlab/data/validate.py`: required-field, severity, LGA, and year-window checks; `validation_{profile}.json` manifests.
- Implemented `src/crashlab/data/clean.py`: PDO rejection, coordinate/hour/duplicate filters, alias normalisation; outputs `brisbane_crashes_cleaned.parquet`, optional `brisbane_crashes_rejected.parquet`, `data_quality_{profile}.json`, `data_quality_summary_{profile}.md`.
- CLI `validate` and `prepare` commands wired; `crashlab all --profile smoke` reaches `status: completed_data_stages` offline with `CRASHLAB_ALLOW_NETWORK=0`.
- Tests: **33 passed** (`pytest tests/`); includes integration `test_smoke_fixture_to_parquet_offline`.
- Observed smoke `--force` runtime: ~0.20s total (acquire ~0.08s, validate ~0.01s, prepare ~0.07s on local Windows run).
