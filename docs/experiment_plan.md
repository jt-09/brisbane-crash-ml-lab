# Experiment plan

## Primary question

How well do CPU-friendly models predict severe Brisbane injury crashes under time-based validation without target leakage?

## Tasks and champions

| Task | Moments / scope | Selection metric (validation) |
|------|-----------------|-------------------------------|
| Binary severity | context, triage | PR-AUC → Brier → recall@top10% |
| Multiclass | context, triage | Macro F1, QWK |
| Ordinal | context, triage (when enabled) | MAE, QWK |
| Anomalies | all injury crashes | Review tables, stability |
| Hotspots | spatial subsets | Cluster coverage, noise fraction |
| Counts | suburb-month | Poisson deviance vs baselines |

`leakage_demo` is excluded from leaderboards.

## Explanation (Phase H)

Bounded on held-out test data:

- Permutation importance (`tuning.permutation_repeats`)
- Partial dependence for top features
- Calibration comparison (none / sigmoid / isotonic)
- FP/FN tables and subgroup metrics (small-*n* flagged)
- Bootstrap CIs for PR-AUC and Brier
- Optional feature ablations (skipped in `smoke`)

All interpretation strings describe **predictive associations**, not causation.

## Profiles

- **smoke** — fixture, &lt;5 min, minimal models and explanation budget.
- **standard** — full Brisbane subset, ≤25 min target.
- **extended** — optional heavier tuning (non-default).
