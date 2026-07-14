# PR 01 — chore: bootstrap cpu-first python project

## Summary

Delivers Phase A bootstrap for Brisbane Crash ML Lab: YAML configuration with profile inheritance, shared logging and path helpers, manifest utilities, a Typer CLI with the full planned command surface, and Makefile targets wired to those commands. Pipeline stages beyond bootstrap planning remain unimplemented until Phase B+. Package version is `0.1.0a1`. No raw data is downloaded in this phase; the pre-seeded local reuse policy is unchanged.

## Changes

- **Configuration (`config.py`, `configs/`)**
  - Profile YAML with `inherits` and deep merge: `smoke` and `standard` inherit `base.yaml`; `extended` inherits `standard.yaml` (and therefore `base.yaml`).
  - Resolved repo-relative paths, config digest hashing, and `CrashlabConfig` dataclass.
- **Infrastructure helpers**
  - `logging.py`: structured console logging and optional JSONL file sink.
  - `paths.py`: repository layout discovery and `ensure_dirs` for data/artifact directories.
  - `data/manifest.py`: run and acquisition manifest read/write helpers (git SHA, UTC timestamps, JSON payloads).
- **CLI (`cli.py`)**
  - Typer entrypoint `crashlab` with `--profile` (`smoke`, `standard`, `extended`) on all pipeline commands.
  - **Implemented:** `version`, `doctor`, `all` (bootstrap plan only for smoke).
  - **Stubbed (raise `NotImplementedError`):** `acquire`, `validate`, `prepare`, `train-binary`, `train-multiclass`, `train-ordinal`, `detect-anomalies`, `cluster-hotspots`, `train-counts`, `report`.
  - `all` logs the ten planned stages and writes a bootstrap manifest with `status: bootstrap_plan_only`.
- **Pipeline (`pipeline.py`)**
  - `run_all` wires config, logging, directories, and stage plan; no acquisition or ML work yet.
- **Makefile**
  - Full target surface: `setup`, `lint`, `typecheck`, `test`, `smoke`, `download`, `validate`, `prepare`, `train`, `anomalies`, `spatial`, `counts`, `report`, `all`, `app`, `clean-generated`, `help`.
  - `make smoke` runs unit tests, `crashlab version`, `crashlab doctor --profile smoke`, and `crashlab all --profile smoke`.
- **Package**
  - Version bumped to `0.1.0a1` in `src/crashlab/__init__.py` and `pyproject.toml`.
  - Editable install via `uv sync`; console script `crashlab` registered.

## Tests

Verified on branch `chore/bootstrap` (2026-07-14):

- [x] `uv run pytest tests/unit -q` — **8 passed** (`test_config.py`, `test_paths.py`, `test_version.py`)
- [x] `uv run crashlab version` — prints `0.1.0a1`
- [x] `uv run crashlab doctor --profile smoke` — loads config, ensures directories, emits JSON health summary
- [x] `uv run crashlab all --profile smoke` — bootstrap plan completes; writes `artifacts/manifests/run_all_smoke_bootstrap.json`

Not run in this verification pass: `make lint`, `make typecheck`, full `make smoke` (individual smoke steps verified separately).

## Metrics/runtime impact

- **Data:** no network download; pre-seeded `data/raw/brisbane_crashes_2015_2023.csv` reuse policy unchanged (`preseed_raw` in `base.yaml`).
- **Smoke profile:** fixture-only (`data/samples/fixture.csv`); `all` is bootstrap-plan-only (sub-second on local machine, ~0.03s observed).
- **Models / training:** none; CPU and memory footprint unchanged from environment setup.
- **Artifacts:** bootstrap manifest JSON only; no model binaries or reports generated.

## Risks

- Makefile targets for `download`, `validate`, `prepare`, training, and `report` invoke CLI commands that raise `NotImplementedError` until Phase B+.
- `standard` and `extended` profiles will attempt real data paths once stages land; clean clones without the gitignored pre-seed must run `acquire` (or copy the local extract).
- Smoke `all` succeeds but does not validate end-to-end ML behaviour; do not treat bootstrap manifest as a training run record.

## Artifacts

- `artifacts/manifests/build_log.md` — Phase A bootstrap notes (updated).
- `artifacts/manifests/run_all_smoke_bootstrap.json` — produced by `crashlab all --profile smoke` (bootstrap plan metadata).
- `artifacts/manifests/cli_smoke.jsonl` — CLI JSONL log from doctor/all smoke invocations (when run).

## Checklist

- [x] Config profiles inherit correctly (`smoke` / `standard` → `base`; `extended` → `standard`)
- [x] Logging, paths, and manifest helpers in place
- [x] Typer CLI exposes full planned command surface
- [x] `version`, `doctor`, and bootstrap `all` work; other stages deferred to Phase B+
- [x] Makefile targets documented and wired
- [x] Unit tests pass (8)
- [x] Version `0.1.0a1`
- [x] No raw government data committed
- [x] No causal or operational road-safety claims in code or docs
- [x] Queensland / OpenDataSoft attribution and CC BY 4.0 preserved in `base.yaml`
