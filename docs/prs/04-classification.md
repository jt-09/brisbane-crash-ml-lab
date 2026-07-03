# PR 04 — feat(classification): binary, multiclass, and ordinal severity models

## Summary

Delivers Phase E classification on branch `feat/classification`. Binary baselines (dummy, logistic, calibrated variants) train per feature moment with champion selection on validation PR-AUC. Multiclass and ordinal experiments extend the same pipeline with macro-F1 / QWK metrics and cumulative-link ordinal models.

## Changes

- **Shared training (`src/crashlab/models/common.py`)**
  - Leaderboard helpers, champion selection, leakage_demo exclusion from production picks.
- **Binary (`src/crashlab/models/binary.py`)**
  - Context/triage/leakage_demo moments; calibrated and uncalibrated baselines; JSON manifests.
- **Multiclass / ordinal (`src/crashlab/models/multiclass.py`, `ordinal.py`)**
  - Four-class severity models; ordinal cumulative probabilities with monotonicity enforcement.
- **Evaluation (`src/crashlab/evaluation/classification.py`, `calibration.py`)**
  - PR-AUC, recall@top-risk, Brier, calibration curves, per-class and QWK metrics.
- **CLI / pipeline**
  - `train-binary`, `train-multiclass`, `train-ordinal` commands wired into `run_all` (ordinal skipped when disabled).

## Tests

- `tests/unit/test_champion_selection.py` — champion picks and leakage_demo exclusion.
- `tests/unit/test_classification_metrics.py` — metric helpers and calibration utilities.
- `tests/integration/test_smoke_train_binary.py` — prepare → train-binary smoke path.
- `tests/unit/test_smoke_training.py` — fixture feature build with binary and multiclass smoke.

## Risks

- **Leakage_demo moment:** pedagogy-only; never promoted to champion.
- **Class imbalance:** PR-AUC and recall@top-risk prioritised over accuracy; review per deployment context.
- **Ordinal link:** simplified cumulative model; not a causal severity ordering claim.

## Checklist

- [x] Binary baselines with calibration and champion selection
- [x] Multiclass and ordinal experiments behind CLI/pipeline stages
- [x] Metric helpers unit-tested
- [x] Smoke integration on fixture profile
- [x] No raw government data or model binaries committed
- [x] Association language only; no operational road-safety claims
