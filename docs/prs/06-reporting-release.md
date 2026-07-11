# PR 06 — feat(reporting-release): evaluation reports, Streamlit explorer, and release docs

## Summary

Delivers Phase H/I reporting on branch `feat/reporting-release`. Static HTML reports aggregate pipeline artifacts with explanation budgets and error analysis; a read-only Streamlit app explores metrics and figures. Release documentation finalises model card, provenance, and v0.1.0 checklist.

## Changes

- **Reports (`src/crashlab/evaluation/reports.py`)**
  - `run_report` builds `reports/index.html` from manifests, metrics, EDA, and explanation artifacts.
- **Explanation (`src/crashlab/evaluation/explanation.py`)**
  - Permutation importance, partial dependence, and bootstrap stability within config budgets.
- **Error analysis (`src/crashlab/evaluation/error_analysis.py`)**
  - Confusion slices, subgroup metrics, and calibration comparison helpers.
- **Artifact loader (`src/crashlab/evaluation/artifact_loader.py`)**
  - Shared JSON/figure discovery for report and app layers.
- **Streamlit app (`src/crashlab/app/streamlit_app.py`)**
  - Read-only explorer over smoke/standard artifacts; no training or inference.
- **CLI / pipeline**
  - `report` stage wired into `crashlab report` and `run_all`.
- **Configs**
  - Explanation budgets (`permutation_repeats`, `bootstrap_samples`, `pdp_max_features`, etc.) in smoke/standard profiles.
- **Docs**
  - Architecture, data dictionary, provenance, experiment plan, model card, release checklist, README quickstart.

## Tests

- `tests/unit/test_reports.py`, `test_error_analysis.py`, `test_streamlit_app.py` — unit coverage for report and app helpers.
- `tests/integration/test_smoke_all_report.py` — smoke `all` path including report stage.

## Risks

- **Explanation budgets:** smoke profile truncates permutation/PDP work; standard may still be slow on low-end CPUs.
- **Streamlit:** displays precomputed artifacts only; stale UI if pipeline not rerun after config changes.
- **Interpretation:** association language only; no operational road-safety or causal claims.

## Checklist

- [x] Static HTML report from pipeline artifacts
- [x] Explanation and error-analysis modules with config budgets
- [x] Streamlit read-only explorer
- [x] CLI and pipeline `report` stage wired
- [x] Model card, provenance, and release checklist documented
- [x] Smoke integration including report stage
- [x] No raw government data or model binaries committed
- [x] Association language only; no operational road-safety claims
