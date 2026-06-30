# PR 03 — feat(features): add temporal splits and leakage-safe features

## Summary

Delivers Phase D (EDA and feature engineering) on branch `feat/eda-features`. Whole-year train/validation/test splits are configured in `configs/base.yaml` (train through 2021, val 2022, test 2023). Feature builders produce context, triage, and leakage-demo moment matrices with train-only encoders, spatial cells, cyclic time features, and a enforced leakage denylist. EDA generators write reproducible tables and figures; notebook stubs delegate to package code.

## Changes

- **Splits (`src/crashlab/features/temporal.py`)**
  - `YearSplits` with disjoint whole-year assignment; proportional fallback for short year spans.
  - Cyclic hour features, weekend flag, and speed buckets.
- **Feature policy (`src/crashlab/features/constants.py`)**
  - Leakage denylist (casualty counts, severity labels, targets), identifier exclusions, context vs triage column groups, and `leakage_demo` extras for pedagogy.
- **Encoders (`src/crashlab/features/encoders.py`)**
  - Train-only `EncoderBundle` with mixed numeric / one-hot encoding; unknown categories mapped at transform time.
- **Builders (`src/crashlab/features/build.py`, `targets.py`, `spatial.py`)**
  - `run_feature_build` for each moment; Parquet matrices per split, joblib encoder artifacts, JSON manifest.
- **EDA (`src/crashlab/evaluation/eda.py`)**
  - Year/severity tables, spatial coverage, split counts; matplotlib figures to `reports/figures/`.
- **Data helpers (`src/crashlab/data/artifacts.py`, `clean.py`)**
  - Processed-path helper; minor clean/prepare hooks for feature pipeline integration.
- **Notebooks (`notebooks/01_data_audit.ipynb`, `02_eda.ipynb`, `03_error_analysis.ipynb`)**
  - Thin stubs calling package entry points.
- **Config (`configs/base.yaml`)**
  - `splits.train_year_end`, `val_years`, `test_years`.

## Tests

- `tests/unit/test_splits.py` — disjoint year splits, split-column assignment, train-only encoder fit.
- `tests/unit/test_leakage.py` — denylist blocks casualty/severity columns in valid moments; leakage_demo allows pedagogy columns.
- `tests/integration/test_smoke_prepare.py` — extended for feature-stage smoke coverage.

## Risks

- **Leakage_demo moment:** casualty columns are intentionally allowed for teaching; must never be used in production severity models.
- **Encoder unknowns:** unseen categories at transform map to a reference level; monitor cardinality in EDA before modelling.
- **Spatial cells:** coarse grid bucketing; resolution is a modelling choice, not a safety claim.

## Checklist

- [x] Whole-year splits configured and tested
- [x] Context / triage / leakage_demo builders with manifests
- [x] Train-only encoders
- [x] Leakage denylist enforced by tests
- [x] EDA generators and notebook stubs
- [x] No raw government data committed
- [x] Association language only; no causal road-safety claims
