# PR 02 — feat(data): add filtered crash data acquisition

## Summary

Delivers Phase B (filtered crash data acquisition) and Phase C (validate / clean / prepare) on branch `feat/data-acquisition`. Validation and cleaning landed in the same branch for delivery efficiency so smoke and standard profiles can run acquire → validate → prepare end to end before Phase D modelling.

Raw Brisbane crash records are sourced from Queensland OpenDataSoft (Crash data from Queensland roads, CC BY 4.0). The pipeline prefers a local pre-seeded extract when its SHA-256 matches the configured digest; otherwise it downloads a filtered CSV (Brisbane City LGA, 2015–2023) with adaptive year selection. Smoke runs stay offline via `CRASHLAB_ALLOW_NETWORK=0` and write to a dedicated fixture raw path that never overwrites the preseed file.

## Changes

- **Schema (`src/crashlab/data/schema.py`)**
  - Canonical field list, OpenDataSoft alias normalisation, severity vocabulary (PDO excluded from modelling set), Brisbane LGA constant, and pinned preseed SHA-256 `CE2A0435366E0870F238295CB9A2A700C12F6FFB25815B8D51ACA2A506CEFF22`.
- **Acquisition (`src/crashlab/data/acquire.py`)**
  - OpenDataSoft export with `select` / `where` filters, metadata-driven field mapping, tenacity retries, and acquisition manifests.
  - Preseed short-circuit when `data/raw/brisbane_crashes_2015_2023.csv` matches configured SHA (31,046 rows, ~9.45 MiB per `configs/base.yaml`).
  - Fixture mode copies `data/samples/fixture.csv` to `data/raw/fixture_smoke.csv` (smoke profile); guards refuse writes to the preseed basename or any path whose SHA matches the preseed digest.
  - `CRASHLAB_ALLOW_NETWORK=0` blocks network download (CI / offline); raises when no reusable preseed is present.
- **Validation (`src/crashlab/data/validate.py`)**
  - Contract checks on required columns, severity vocabulary, LGA filter, and year window; writes `artifacts/manifests/validation_{profile}.json`.
- **Cleaning / prepare (`src/crashlab/data/clean.py`)**
  - Rejects PDO rows, invalid coordinates / hours, duplicates, and out-of-window years; normalises aliases and severity labels.
  - Outputs cleaned Parquet, optional rejected Parquet, quality JSON manifest, and Markdown summary report.
- **CLI and pipeline (`cli.py`, `pipeline.py`)**
  - `acquire`, `validate`, and `prepare` commands implemented; `all` runs the three data stages and records `status: completed_data_stages`.
  - Training and reporting stages remain stubbed for Phase D+.
- **Configuration (`configs/smoke.yaml`)**
  - `fixture_raw_path: data/raw/fixture_smoke.csv` documents the dedicated smoke destination separate from `preseed_raw.path`.

## Tests

Verified on branch `feat/data-acquisition` (2026-07-14):

- [x] `.venv\Scripts\python.exe -m pytest tests/ -q` — **33 passed**
  - `tests/unit/test_schema.py` (7)
  - `tests/unit/test_acquire_preseed.py` (6)
  - `tests/unit/test_acquire_network.py` (5)
  - `tests/unit/test_clean.py` (6)
  - `tests/unit/test_config.py` (5)
  - `tests/unit/test_paths.py` (2)
  - `tests/unit/test_version.py` (1)
  - `tests/integration/test_smoke_prepare.py` (1)
- [x] `CRASHLAB_ALLOW_NETWORK=0 crashlab all --profile smoke --force` — acquire, validate, prepare complete offline; preseed file untouched
- [x] Preseed isolation tests assert smoke fixture acquisition does not modify `brisbane_crashes_2015_2023.csv` mtime or SHA when present

Not run in this verification pass: `make lint`, `make typecheck`, live network acquisition against OpenDataSoft.

## Metrics / runtime

- **Preseed reuse (standard profile):** SHA-256 `CE2A0435…CEFF22`, 31,046 rows, 9,907,354 bytes (~9.45 MiB); no download when hash matches.
- **Smoke profile (`--force`, offline):** 20 fixture rows; acquire ~0.08s, validate ~0.01s, prepare ~0.07s, total ~0.20s (local Windows run, 2026-07-14).
- **Smoke raw output:** `data/raw/fixture_smoke.csv` (6,352 bytes observed); distinct from preseed path.
- **Cleaned output (smoke):** `data/processed/brisbane_crashes_cleaned.parquet` (20 rows, 0 rejected, PDO excluded from severity distribution).
- **Models / training:** none; CPU-only data stages only.

## Risks

- **Smoke path isolation:** Early smoke runs briefly wrote fixture content onto the preseed CSV; the preseed was restored from OpenDataSoft with matching SHA `CE2A0435…CEFF22`, and acquisition guards now refuse preseed overwrites (basename check plus SHA match). Reviewers should confirm `fixture_raw_path` stays separate from `preseed_raw.path`.
- **Preseed SHA drift:** If the local extract is replaced or corrupted, `CRASHLAB_ALLOW_NETWORK=0` fails fast; standard acquisition requires network or a matching preseed. Document any upstream Queensland data refresh and update `preseed_raw.sha256` deliberately.
- **Network acquisition:** Live OpenDataSoft export size and row counts depend on upstream filters; adaptive year expansion is bounded by `max_raw_bytes` in config.
- **Modelling leakage:** Cleaning removes PDO and invalid rows but does not implement train/test splits; Phase D must enforce time-based whole-year validation and the leakage denylist.

## Artifacts

| Path | Description |
|------|-------------|
| `data/raw/fixture_smoke.csv` | Smoke acquisition output (gitignored) |
| `data/raw/brisbane_crashes_2015_2023.csv` | Preseed standard raw (gitignored) |
| `data/processed/brisbane_crashes_cleaned.parquet` | Cleaned modelling-ready table |
| `data/interim/brisbane_crashes_rejected.parquet` | Rejected rows (written when rejects exist) |
| `artifacts/manifests/acquisition_{profile}.json` | Acquisition provenance and timings |
| `artifacts/manifests/validation_{profile}.json` | Validation contract report |
| `artifacts/manifests/data_quality_{profile}.json` | Row counts, rejection reasons, cardinality |
| `reports/data_quality_summary_{profile}.md` | Human-readable quality summary |
| `artifacts/manifests/run_all_{profile}.json` | End-to-end stage results for `crashlab all` |

Data attribution: Queensland Government crash data via [OpenDataSoft](https://queensland.opendatasoft.com/explore/dataset/road-crash-locations-queensland/) and [data.qld.gov.au](https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads), licensed CC BY 4.0.

## Checklist

- [x] Schema contract and preseed SHA constant in place
- [x] Acquisition: network, preseed reuse, and fixture modes
- [x] `CRASHLAB_ALLOW_NETWORK=0` honoured in acquire and CI
- [x] Smoke writes `fixture_smoke.csv`; preseed clobber guards enforced
- [x] Validation manifest and contract checks
- [x] Prepare writes cleaned Parquet, rejected rows, quality JSON/MD
- [x] CLI `acquire`, `validate`, `prepare`, and data-stage `all` wired
- [x] Unit and integration tests pass (33)
- [x] No raw government data committed
- [x] Association / prediction language only; no causal road-safety claims
- [x] Queensland / OpenDataSoft attribution and CC BY 4.0 preserved in config
