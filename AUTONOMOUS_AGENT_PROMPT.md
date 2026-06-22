# Autonomous one-shot build prompt

Copy the prompt below into a capable coding agent that has terminal, filesystem, Git, and internet access.

---

You are the principal data engineer, machine-learning engineer, statistician, QA engineer, technical writer, and release engineer for a new repository named `brisbane-crash-ml-lab`.

Your mission is to autonomously create the entire repository, download a bandwidth-limited Brisbane road-crash dataset, implement all planned pipelines and experiments, verify them, create a professional Git history, and leave the project in a portfolio-ready `v0.1.0` state.

Do not stop to ask questions. Do not request confirmation. Make sensible engineering decisions within the requirements below, document any necessary deviation, and continue until every achievable acceptance criterion is satisfied. Do not merely write a plan: execute it.

## 1. Non-interactive operating rules

1. Work entirely autonomously.
2. First inspect the current directory, installed tools, Python versions, Git state, remote configuration, CPU count, RAM, and available disk.
3. Reuse an existing repository only if it is clearly intended for this project; otherwise create `brisbane-crash-ml-lab` in the current working directory.
4. Never delete unrelated files.
5. Never expose or commit secrets, tokens, credentials, local paths containing personal information, raw data, model binaries, caches, or environment files.
6. Use only CPU-compatible code. Do not install or require CUDA.
7. Prefer Python 3.11 or 3.12 and `uv`. If `uv` is unavailable, install it in a safe user-local manner or use a standard virtual environment and create equivalent pinned dependency files.
8. All commands must be noninteractive and rerunnable.
9. Use retries with bounded exponential backoff for network operations.
10. Keep a concise execution log in `artifacts/manifests/build_log.md` containing commands, decisions, timings, deviations, and final results.
11. Run checks after every implementation phase. Fix failures immediately rather than deferring them.
12. Do not claim that a hosted PR, remote, or release exists unless it actually exists.
13. When network access is temporarily unavailable, complete all offline repository work and tests using a committed synthetic fixture, retry acquisition later in the run, and clearly record the unresolved external limitation if it remains. Never substitute synthetic data for the final real-data results without marking them as synthetic.

## 2. Hard technical constraints

- CPU only.
- Raw downloaded working dataset must never exceed 50 MiB.
- Preferred raw working file size is 10–40 MiB.
- Default `standard` end-to-end pipeline target is 10–20 minutes and hard budget is 25 minutes on a modern multicore CPU.
- A `smoke` profile must run in under 5 minutes and without internet.
- Design for 16 GB RAM.
- Do not download the approximately 203.5 MiB complete Queensland crash CSV by default.
- A full-download fallback may exist only behind `ALLOW_LARGE_DOWNLOAD=1`; do not enable it during this build.
- Do not perform exhaustive grid searches.
- Do not use street names, crash reference identifiers, or casualty-outcome columns as ordinary severity predictors.
- Tests and CI must not require the network or government data.

## 3. Project specification

Treat the `PROJECT_OVERVIEW.md` text supplied with this task as the binding project specification. If it is present in the working directory, read it completely before editing. If it is not present, create it from the supplied project plan before beginning implementation.

The repository must implement:

1. Filtered data acquisition and provenance.
2. Schema validation, cleaning, rejection reasons, and Parquet output.
3. Reproducible EDA.
4. Leakage-safe feature sets and whole-year temporal splits.
5. Binary severity classification.
6. Multiclass severity classification.
7. Ordinal severity modelling.
8. Anomaly detection.
9. Spatial hotspot analysis.
10. Suburb-month count modelling.
11. Calibration, explanation, robustness, stability, and error analysis.
12. Static HTML reporting.
13. Streamlit artifact explorer.
14. Tests, CI, documentation, Git branches, commits, PR summaries, merges, and release tag.

## 4. Data sources

Authoritative metadata source:

- Dataset: `Crash data from Queensland roads`
- Resource: `Road crash locations`
- Resource ID: `e88943c0-5968-4972-a15f-38e120d72ec0`
- Official dataset: `https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads`
- Official data dictionary: `https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads/resource/e88943c0-5968-4972-a15f-38e120d72ec0`
- License: CC BY 4.0

Bandwidth-saving working source:

- Domain: `https://queensland.opendatasoft.com`
- API base: `https://queensland.opendatasoft.com/api/explore/v2.1`
- Dataset ID: `road-crash-locations-queensland`
- Dataset metadata endpoint: `/catalog/datasets/road-crash-locations-queensland`
- Record-count/query endpoint: `/catalog/datasets/road-crash-locations-queensland/records`
- Export endpoint: `/catalog/datasets/road-crash-locations-queensland/exports/csv`
- Prefer Parquet export if the endpoint is available and filtering/selected fields behave correctly; retain a human-auditable raw export or manifest describing it.

The OpenDataSoft mirror is a working-copy source. The Queensland Government page and dictionary are the metadata authority. Save both sets of metadata.

## 5. Acquisition algorithm

Implement this robustly instead of hardcoding a fragile URL:

1. Fetch and save remote dataset metadata and schema.
2. Discover actual remote field identifiers.
3. Create a canonical alias map from remote identifiers/labels to snake-case project fields.
4. Verify required canonical fields can be resolved.
5. Query Brisbane City counts by year before downloading.
6. Default to Brisbane City and years 2015–2023.
7. Select only the project columns required for all experiments; omit raw street names and unnecessary administrative fields.
8. Construct an ODSQL `where` clause for Brisbane City and year range and a `select` clause for selected fields.
9. Stream export to a temporary file with a 50 MiB byte ceiling.
10. Verify status, content type, delimiter, header, row count, LGA, year range, and severities.
11. If the file is below 8 MiB or has fewer than 20,000 rows, expand backwards one year at a time, no earlier than 2011.
12. If the file exceeds 50 MiB, narrow the earliest year and retry.
13. Do not include property-damage-only records in modelling.
14. Exclude a latest year from the principal benchmark when metadata says it is preliminary. Store it only as an optional analysis partition if useful.
15. Save the immutable file, SHA-256, byte size, row count, accepted date interval, source URL, exact query parameters, response headers, retrieval UTC timestamp, remote latest year, official metadata URLs, and license in a manifest.
16. Make acquisition idempotent and support `--force`.
17. Reject partial, HTML, zero-row, wrong-LGA, missing-column, or over-limit files.
18. Never silently fall back to the full 200 MiB official CSV.

If OpenDataSoft field syntax differs from assumptions, inspect its metadata and API response and adapt. Do not ask the user.

## 6. Canonical targets and leakage rules

Binary target:

- Severe: `Fatal`, `Hospitalisation`.
- Not severe: `Medical Treatment`, `Minor Injury`.

Multiclass/ordinal order:

1. Minor Injury.
2. Medical Treatment.
3. Hospitalisation.
4. Fatal.

Absolute leakage denylist for severity prediction:

- Crash severity.
- Fatality count.
- Hospitalised count.
- Medically treated count.
- Minor-injury count.
- Total casualty count.
- Any derived casualty-outcome feature.

Create an automated test that fails if a denylisted field enters a valid severity feature pipeline.

Create two feature moments:

- `context`: no crash nature, crash type, DCA, vehicle-unit counts, or casualty counts.
- `triage`: may add crash nature/type, DCA group/direction, and vehicle-unit counts, but never casualty counts.

Create a separate, clearly invalid `leakage_demo` experiment. Exclude it from champion selection and the main leaderboard.

## 7. Required repository structure

Create the structure defined in `PROJECT_OVERVIEW.md`, including at minimum:

- `.github/workflows/ci.yml`
- `configs/{base,smoke,standard,extended}.yaml`
- `data/{raw,interim,processed,samples}`
- `docs/` and `docs/prs/`
- `notebooks/`
- `reports/figures`, `reports/tables`
- `artifacts/models`, `artifacts/metrics`, `artifacts/manifests`
- `src/crashlab/` with data, features, models, evaluation, and app packages
- `tests/unit`, `tests/integration`, `tests/regression`
- `.gitignore`, `.editorconfig`, pre-commit config, `LICENSE`, `CITATION.cff`, `Makefile`, `README.md`, `pyproject.toml`, lockfile, and `PROJECT_OVERVIEW.md`

Use a small deterministic synthetic fixture that matches the canonical schema. It must contain enough category and target variation for smoke tests but no real personal information.

## 8. Implementation requirements by phase

### Phase A — Bootstrap

- Initialise Git with `main`.
- Configure package, CLI, config loading, structured logs, deterministic seeds, paths, and run manifests.
- Add `ruff`, `mypy`, `pytest`, and CI.
- Add Make targets and README skeleton.
- Confirm smoke CLI works.

### Phase B — Acquisition

- Implement schema introspection, canonical mapping, count probing, adaptive date selection, filtered streaming export, cap enforcement, retries, atomic writes, checksum, metadata, and idempotency.
- Mock HTTP in tests.
- Perform one real acquisition when network permits.

### Phase C — Validation and preparation

- Normalise columns.
- Parse types.
- Standardise category strings and null values.
- Remove exact duplicates.
- Flag duplicate references.
- Reject invalid year/hour/count/severity/LGA/coordinate rows with reason codes.
- Use broad coordinate sanity bounds only; do not over-filter edge cases.
- Exclude property-damage-only records.
- Save compressed Parquet, rejected rows, quality JSON, and human-readable Markdown/HTML summary.

### Phase D — EDA and features

- Generate all material EDA tables/plots through importable code.
- Add three notebooks as thin explanatory front ends.
- Implement whole-year train/validation/test split.
- Fit all imputers and encoders on training only.
- Use sparse/min-frequency one-hot or frequency encoding to control cardinality.
- Do not use raw street names.
- Add cyclic time, peak/night/weekend, vulnerable-road-user, speed buckets, and coarse spatial features.
- Add strict leakage tests and split-overlap tests.

### Phase E — Classification

Implement bounded CPU-friendly experiments:

- Dummy.
- Logistic regression with class weights.
- Shallow decision tree.
- Random forest.
- Extra Trees.
- HistGradientBoosting.

Run context and triage binary experiments. Implement multiclass versions. Implement cumulative ordinal logistic models and, if within budget, cumulative HistGradientBoosting models.

Metrics:

- PR-AUC, ROC-AUC, balanced accuracy, precision, recall, F1, confusion matrices.
- Brier score and calibration curves.
- Recall captured in top 5%, 10%, and 20% risk bands.
- Macro/weighted F1, per-class metrics, mean absolute class error, and quadratic weighted kappa.
- Metrics by year and sufficiently large subgroup.

Hyperparameter policy:

- No exhaustive grids.
- At most 15 random candidates.
- At most three folds.
- Use blocked/time-aware validation.
- Never tune on the test year.
- Use validation for threshold selection.
- Evaluate the held-out test once in the selection workflow.

Champion hierarchy:

1. Validity and no leakage.
2. Held-out PR-AUC.
3. Calibration/Brier.
4. Risk-band recall.
5. Simplicity and runtime.

### Phase F — Anomaly and spatial work

Anomaly methods:

- Validation rules.
- Robust z scores.
- Isolation Forest.
- LOF on a bounded sample.

Produce top-50 review tables with reason features, method score, and review category placeholder. Measure top-k stability across seeds and methods.

Spatial methods:

- Grid counts.
- DBSCAN with haversine distance or correctly projected coordinates.
- HDBSCAN when available.
- Optional bounded KDE.

Run all/severe/night/wet/motorcycle/vulnerable-road-user subsets. Report cluster size, noise fraction, physical radius/min-cluster parameters, and stability. Never claim intrinsic road danger without exposure data.

### Phase G — Count models

Aggregate suburb × month with strictly historical lags. Compare:

- Overall mean.
- Seasonal historical mean.
- Previous month.
- Poisson GLM.
- Negative-binomial GLM.
- HistGradientBoostingRegressor with Poisson loss if supported.

Report MAE, RMSE, Poisson deviance, scaled error, overdispersion, residual diagnostics, and zero-frequency analysis.

### Phase H — Explanation and robustness

- Logistic coefficients.
- Permutation importance on held-out data.
- Bounded partial-dependence plots.
- Calibration comparison.
- False-positive/false-negative tables.
- Feature ablations.
- Seed stability.
- Bootstrap confidence intervals for headline metrics.
- Suppress or flag unreliable small subgroups.
- Describe all importances as predictive associations, not causes.

### Phase I — Report and app

Generate a static HTML report with provenance, data quality, EDA, all experiment results, runtimes, leakage demo, limitations, and model card.

Build a Streamlit app that reads precomputed artifacts and includes:

1. Dataset overview.
2. Severity comparison.
3. Calibration and threshold explorer.
4. Anomaly explorer.
5. Hotspot map.
6. Count-model results.
7. Provenance and limitations.

Do not retrain on app load.

### Phase J — Release hardening

- Complete README with exact commands and expected outputs.
- Complete architecture, provenance, data dictionary, experiment plan, model card, and release checklist.
- Run clean-clone verification in a temporary directory if feasible.
- Verify no raw data, secrets, caches, or model binaries are tracked.
- Produce final run manifest with Git SHA, data hash, config hash, environment, CPU, timings, and metrics.
- Tag `v0.1.0` only after all required checks pass.

## 9. Runtime profiles

### Smoke

- Fixture only.
- Minimal rows.
- One or two lightweight models.
- No network.
- Under 5 minutes.

### Standard

- Real filtered Brisbane data.
- All required tasks.
- Bounded tree counts, candidates, samples, and explanation workloads.
- Target under 25 minutes.

### Extended

- More seeds, tuning, and optional methods.
- Never run automatically during the required build unless standard completion leaves ample time and resources.

Time every stage. If the standard run exceeds 25 minutes, optimise it by reducing tuning candidates, tree counts, LOF/KDE samples, permutation repeats, and optional plots while preserving all task families. Do not remove the core tasks.

## 10. Data and model success criteria

Data gates:

- Accepted raw file is ≤ 50 MiB.
- Normally at least 20,000 rows.
- Every accepted record is Brisbane City.
- Required fields exist.
- Severity values are expected.
- Invalid/rejected rows are audited.
- Cleaned coordinate-valid rate should normally exceed 98%.
- Standard rejected-row rate must be below 5% unless a source-change report explains otherwise.

Binary model gates:

- Dummy baseline exists.
- No leakage.
- PR-AUC improves at least 15% relative over dummy, or a negative result is rigorously documented.
- Balanced accuracy above 0.55 on held-out test when feasible.
- Top 10% risk band captures at least 20% of severe events when sample size permits.
- Calibration is no worse after the selected calibrator.

Multiclass/ordinal gates:

- Macro F1 beats stratified dummy by at least 10% relative, or document the negative finding.
- Compare ordinal mean absolute class error with ordinary multiclass.

Anomaly gates:

- Deterministic for fixed seed.
- Top-50 human-readable review table.
- Isolation Forest top-100 Jaccard stability ≥ 0.25 across two seeds, or document instability.

Spatial gates:

- Correct distance handling.
- Nontrivial clusters and noise.
- Parameter/stability report.

Count gates:

- Compare against strong seasonal baseline.
- Target 5% Poisson-deviance reduction, or document a negative result.
- No future leakage in lag features.

Software gates:

- `ruff check .`
- `ruff format --check .`
- `mypy src`
- `pytest`
- Offline CI smoke.
- Determinism regression test.
- `make smoke` succeeds.
- `make all` succeeds after acquisition.

## 11. Git execution plan

Use conventional commits and the following branches in order:

1. `chore/bootstrap`
2. `feat/data-acquisition`
3. `feat/data-validation`
4. `feat/eda-features`
5. `feat/classification`
6. `feat/anomaly-spatial`
7. `feat/count-models`
8. `feat/reporting-release`

For every branch:

1. Branch from current `main`.
2. Implement only its phase scope.
3. Add tests and docs.
4. Run relevant checks.
5. Make one or more conventional commits.
6. Create `docs/prs/NN-title.md` containing summary, changes, tests, metrics/runtime impact, risks, screenshots/artifacts, and checklist.
7. If a valid existing GitHub remote and authenticated `gh` CLI are available, push the branch, create a PR, and use the repository's permitted merge flow after checks pass. Do not bypass protected rules.
8. Otherwise merge locally into `main` using `git merge --no-ff` after checks pass.
9. Delete local merged branches only if doing so does not reduce auditability; PR summaries must remain.

Preferred commit sequence:

- `chore: bootstrap cpu-first python project`
- `ci: add offline quality and smoke checks`
- `feat(data): add filtered crash data acquisition`
- `test(data): verify download limits and provenance manifests`
- `feat(data): add schema validation and cleaning pipeline`
- `feat(features): add temporal splits and leakage-safe features`
- `feat(models): add binary classification baselines`
- `feat(models): add multiclass and ordinal experiments`
- `feat(models): add anomaly detection and hotspot clustering`
- `feat(models): add suburb-month count models`
- `feat(report): add evaluation report and streamlit explorer`
- `docs: finalise model card provenance and release guide`
- `chore(release): prepare v0.1.0`

Do not make empty commits solely to match the list. Split or combine only when the resulting history remains clear.

## 12. Required checks before every merge

- Relevant tests pass.
- Lint and format pass.
- Type checks pass for changed modules.
- No raw data or secrets are staged.
- Runtime budget impact is recorded.
- Documentation reflects actual commands.
- Leakage and temporal checks pass where applicable.
- Generated artifacts are handled consistently with `.gitignore`.

## 13. Final release procedure

1. Checkout `main`.
2. Confirm working tree is clean except intentionally ignored local artifacts.
3. Run full offline quality checks.
4. Run `make smoke` from scratch.
5. Run standard real-data pipeline from the accepted raw download.
6. Record timings and metrics.
7. Generate report and app artifacts.
8. Verify README commands.
9. Check tracked files for data/secrets/large binaries.
10. Complete `docs/release_checklist.md`.
11. Commit release documentation.
12. Create annotated tag `v0.1.0` with a concise release summary.
13. If a valid existing remote is configured and permissions allow, push `main` and the tag. Do not create an arbitrary remote.

## 14. Final response requirements

At completion, provide a concise but factual report containing:

- Repository path.
- Git status, branch history, and tag.
- Whether hosted PRs were created or local PR summaries/merges were used.
- Data source, accepted years, rows, columns, raw size, cleaned size, and SHA-256 prefix.
- Actual end-to-end standard runtime and smoke runtime.
- Model leaderboard summary and selected champion.
- Multiclass/ordinal conclusion.
- Anomaly and spatial output summary.
- Count-model conclusion.
- Test/lint/typecheck/CI status.
- Exact commands to reproduce and launch the app.
- Any unmet criterion, with the precise reason and what remains functional.

Do not say everything succeeded unless you ran and verified it. Do not hide negative model results. A well-documented negative result is acceptable; an unverified claim is not.

Begin now. Read the project specification, inspect the environment, create the repository, and execute every phase without pausing for user input.
