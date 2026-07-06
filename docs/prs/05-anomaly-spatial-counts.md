# PR 05 — feat(anomaly-spatial): anomaly detection, hotspots, and count models

## Summary

Delivers Phase F spatial and unsupervised stages on branch `feat/anomaly-spatial`. Anomaly detection flags coordinate and temporal outliers with stability checks; DBSCAN clusters spatial hotspots by severity subset. Suburb-month Poisson/negative-binomial count models complete the pipeline with smoke integration across all three stages.

## Changes

- **Anomalies (`src/crashlab/models/anomalies.py`)**
  - Rule-based and sklearn outlier methods; top-k review table and seed stability via Jaccard overlap.
- **Hotspots (`src/crashlab/models/hotspots.py`)**
  - DBSCAN clustering on lat/lon with severity subsets and cluster summaries.
- **Counts (`src/crashlab/models/counts.py`)**
  - Suburb-month aggregation; Poisson and negative-binomial GLM baselines with deviance metrics.
- **Stability (`src/crashlab/evaluation/stability.py`)**
  - Top-k Jaccard helper for unsupervised reproducibility checks.
- **CLI / pipeline**
  - `detect-anomalies`, `cluster-hotspots`, and `train-counts` wired into `run_all`.

## Tests

- `tests/unit/test_anomalies.py`, `test_hotspots.py`, `test_stability.py`, `test_counts.py` — unit coverage per stage.
- `tests/integration/test_smoke_anomaly_spatial_counts.py` — fixture smoke path for all three stages.

## Risks

- **Spatial clustering:** exploratory association only; not a causal hotspot or intervention claim.
- **Count models:** suburb-month aggregation hides within-suburb heterogeneity; review dispersion assumptions.
- **Anomaly flags:** rule thresholds are heuristic; manual review table is advisory, not ground truth.

## Checklist

- [x] Anomaly detection with stability metrics
- [x] Hotspot clustering with subset summaries
- [x] Suburb-month count baselines
- [x] CLI and pipeline stages wired
- [x] Smoke integration on fixture profile
- [x] No raw government data or model binaries committed
- [x] Association language only; no operational road-safety claims
