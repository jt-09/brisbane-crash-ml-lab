# Architecture

Brisbane Crash ML Lab (`crashlab`) is a CPU-first pipeline around Queensland Brisbane City road-crash records.

## Layers

1. **Data** — `crashlab.data`: acquire, validate, clean, manifests.
2. **Features** — `crashlab.features`: temporal/spatial encoders, leakage-safe moment matrices (`context`, `triage`, `leakage_demo`).
3. **Models** — `crashlab.models`: binary/multiclass/ordinal severity, anomalies, hotspots, counts.
4. **Evaluation** — `crashlab.evaluation`: metrics, calibration, EDA, explanation, static HTML report.
5. **App** — `crashlab.app.streamlit_app`: read-only explorer over artifacts.

## Orchestration

- CLI: `crashlab` (Typer) — one command per stage plus `all`.
- Pipeline: `crashlab.pipeline.run_all` runs stages in order; does not retrain during `report`.
- Config: YAML profiles in `configs/` with inheritance (`smoke`, `standard`, `extended`).

## Artifacts

| Path | Contents |
|------|----------|
| `data/processed/` | Cleaned Parquet |
| `artifacts/models/` | Serialized models (gitignored) |
| `artifacts/metrics/` | JSON metrics and explanation |
| `artifacts/manifests/` | Run manifests |
| `reports/` | HTML report, figures, tables |

## Validation policy

Time-based whole-year splits (`train` ≤ 2021, `val` 2022, `test` 2023 by default). Leakage denylist enforced in feature build and tests.
