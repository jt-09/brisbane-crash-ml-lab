# Brisbane Road Crash ML Lab

> A CPU-first, reproducible machine-learning and data-engineering project using Queensland road-crash data for Brisbane City.

## 1. Project summary

The project builds a compact but complete local ML system around Brisbane road-crash records. It is designed for a machine with a strong CPU, no GPU, and a preference for classical/statistical ML rather than deep learning.

The project deliberately supports several problem types over one governed dataset:

1. Binary crash-severity classification.
2. Multiclass and ordinal severity modelling.
3. Unsupervised anomaly and outlier detection.
4. Spatial hotspot clustering.
5. Suburb-month crash-count regression.
6. Model explanation, calibration, error analysis, and reproducibility checks.

The main output is not merely a notebook. It is a versioned repository with a data-acquisition command, validation layer, feature pipeline, experiment runner, tests, reports, model artifacts, and a small interactive application.

---

## 2. Hard constraints

| Constraint | Requirement |
|---|---|
| Compute | CPU only; no CUDA or GPU assumptions |
| Raw local download | Hard cap: 50 MiB; preferred working range: 10–40 MiB |
| Default end-to-end runtime | Target: 10–20 minutes; hard budget: 25 minutes on a modern multicore CPU |
| Standard model training | Each individual default model should normally complete within 5 minutes |
| Memory | Design for 16 GB RAM; avoid dense high-cardinality matrices |
| Reproducibility | Fixed seeds, pinned environment, immutable raw file, checksums, deterministic splits |
| Data scope | Brisbane City, recent complete historical period, selected columns only |
| Validation | Time-based holdout; random split may be used only as a diagnostic comparison |
| Safety of conclusions | Association and prediction only; no causal or operational road-safety claims |

A `smoke` profile must complete in less than 5 minutes on a normal laptop. A `standard` profile must be engineered to remain within 25 minutes. An optional `extended` profile may exceed the budget but must never be the default.

---

## 3. Data source and provenance

### 3.1 Authoritative source

The authoritative metadata and schema source is the Queensland Government Open Data Portal:

- Dataset: `Crash data from Queensland roads`
- Resource: `Road crash locations`
- Resource ID: `e88943c0-5968-4972-a15f-38e120d72ec0`
- License: Creative Commons Attribution 4.0
- Official dataset page: <https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads>
- Official resource/data dictionary: <https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads/resource/e88943c0-5968-4972-a15f-38e120d72ec0>

At the time this plan was prepared, the official resource described casualty crash locations from 1 January 2001 to 30 June 2025, stated that records from the most recent 12 months may be preliminary, used GDA2020 coordinates, and listed the complete CSV at approximately 203.5 MiB. The complete file is intentionally not the default acquisition path.

### 3.2 Small-download working source

Use the Queensland OpenDataSoft mirror to request only Brisbane City rows, selected years, and selected fields:

- Dataset identifier: `road-crash-locations-queensland`
- Dataset page: <https://queensland.opendatasoft.com/explore/dataset/road-crash-locations-queensland/>
- API base: `https://queensland.opendatasoft.com/api/explore/v2.1`

The OpenDataSoft API supports `where` and `select` clauses and unrestricted export endpoints. The acquisition code must first inspect the remote schema, map remote field identifiers to canonical project names, and then construct the filtered export. Do not assume field-case or field identifiers without introspection.

### 3.3 Source-of-truth rule

The OpenDataSoft export is a bandwidth-saving working copy, not the metadata authority. The pipeline must:

1. Save source URLs and retrieval timestamp.
2. Save remote dataset metadata and schema alongside the data.
3. Verify expected fields against the official Queensland data dictionary.
4. Record the mirror's latest available year.
5. Exclude preliminary years from the principal benchmark unless explicitly enabled.
6. Fail clearly if the mirror is empty, unexpectedly small, missing required fields, or materially stale relative to the configured policy.

### 3.4 Default subset policy

Start with:

- Local government area: `Brisbane City`
- Preferred years: 2015–2023
- Expand backwards as far as 2011 if the selected file is smaller than the preferred minimum size and runtime estimates remain acceptable.
- Do not include property-damage-only records in modelling.
- Do not use 2024 or later in the principal benchmark unless the metadata policy marks the year as complete enough for the selected analysis.

Adaptive download behavior:

1. Request 2015–2023 with the canonical selected columns.
2. If the resulting raw file is between 8 and 50 MiB and has at least 20,000 rows, accept it.
3. If it is below 8 MiB or below 20,000 rows, expand backwards one year at a time, no earlier than 2011.
4. If it exceeds 50 MiB, reduce the earliest year until it is within the cap.
5. Never silently download the approximately 200 MiB full Queensland file.
6. A large-download fallback may exist only behind `ALLOW_LARGE_DOWNLOAD=1` and must be disabled by default.

---

## 4. Project questions

### Primary question

How well can standard CPU-friendly models predict whether a reported Brisbane crash is severe, without target leakage and under realistic time-based validation?

### Secondary questions

- Does an ordinal treatment of severity outperform ordinary multiclass classification?
- Which crash records are unusual under mixed temporal, road, location, and vehicle features?
- Do severe crashes form spatial clusters distinct from all-crash clusters?
- Can suburb-month severe-crash counts be modelled better with Poisson or negative-binomial assumptions than with ordinary regression?
- How much predictive value comes from post-incident crash descriptors compared with information available before a crash?
- How stable are conclusions across years, suburbs, feature sets, seeds, and model families?

---

## 5. Prediction moments and leakage policy

A feature is valid only if it would exist at the declared prediction moment.

### 5.1 Model A: contextual/pre-crash model

Purpose: estimate severity risk conditional on a crash occurring, using context that could be known from location, time, and environment.

Candidate features:

- Year, month, day of week, hour.
- Weekend, peak-hour, night, and seasonal indicators.
- Suburb, postcode, SA2, and coarse coordinate cells.
- Roadway feature.
- Traffic control.
- Speed limit.
- Road surface, atmospheric condition, lighting, horizontal alignment, vertical alignment.
- Controlling authority.

Excluded from this model:

- Crash nature and crash type.
- DCA code, DCA description, and DCA group.
- Counts or types of involved units.
- Casualty counts.

### 5.2 Model B: immediate post-incident triage model

Purpose: estimate severity after initial incident details are available.

Model B may add:

- Crash nature and crash type.
- DCA group and approach direction.
- Counts of cars, motorcycles, trucks, buses, bicycles, pedestrians, and other units.

### 5.3 Absolute leakage denylist

Never use these fields as predictors of crash severity:

- `Crash_Severity` itself.
- `Count_Casualty_Fatality`.
- `Count_Casualty_Hospitalised`.
- `Count_Casualty_MedicallyTreated`.
- `Count_Casualty_MinorInjury`.
- `Count_Casualty_Total`.
- Any feature directly derived from the target or casualty outcomes.

The repository must include an automated leakage test that fails if any denylisted field enters a severity-model feature matrix.

### 5.4 Leakage demonstration experiment

For educational purposes only, train a deliberately invalid model that includes casualty counts and compare it with the valid model. Store it under an explicitly named `leakage_demo` experiment. It must never be included in the main leaderboard or exported as a usable model.

---

## 6. Canonical data contract

The acquisition layer may receive different remote field identifiers, but the cleaned dataset must use these canonical names where available:

```text
crash_ref_number
crash_severity
crash_year
crash_month
crash_day_of_week
crash_hour
crash_nature
crash_type
crash_longitude
crash_latitude
loc_suburb
loc_local_government_area
loc_post_code
loc_abs_statistical_area_2
crash_controlling_authority
crash_roadway_feature
crash_traffic_control
crash_speed_limit
crash_road_surface_condition
crash_atmospheric_condition
crash_lighting_condition
crash_road_horiz_align
crash_road_vert_align
crash_dca_code
crash_dca_group_description
dca_key_approach_dir
count_casualty_fatality
count_casualty_hospitalised
count_casualty_medically_treated
count_casualty_minor_injury
count_casualty_total
count_unit_car
count_unit_motorcycle_moped
count_unit_truck
count_unit_bus
count_unit_bicycle
count_unit_pedestrian
count_unit_other
```

Required columns for the minimum viable project:

```text
crash_ref_number
crash_severity
crash_year
crash_month
crash_day_of_week
crash_hour
crash_longitude
crash_latitude
loc_suburb
loc_local_government_area
crash_roadway_feature
crash_traffic_control
crash_speed_limit
crash_road_surface_condition
crash_atmospheric_condition
crash_lighting_condition
crash_road_horiz_align
crash_road_vert_align
```

If a nonessential field is absent, the pipeline should log the omission and disable only the experiments that require it. If a required field is absent, acquisition must fail.

---

## 7. Targets

### 7.1 Binary severity

```text
severe = Fatal or Hospitalisation
not_severe = Medical Treatment or Minor Injury
```

Property-damage-only records are excluded.

### 7.2 Multiclass severity

```text
0 = Minor Injury
1 = Medical Treatment
2 = Hospitalisation
3 = Fatal
```

### 7.3 Ordinal severity

Train cumulative threshold models:

```text
P(severity >= Medical Treatment)
P(severity >= Hospitalisation)
P(severity >= Fatal)
```

Apply probability-consistency correction if independently fitted thresholds cross.

### 7.4 Anomaly task

No supervised target. Produce anomaly scores for valid cleaned records and manually review the top-ranked examples.

### 7.5 Count-regression task

Aggregate to `suburb × year × month` and predict:

- Total crashes.
- Severe crashes.

This is a count model, not a measure of intrinsic road danger. Without traffic exposure, interpretations must remain about recorded counts.

---

## 8. Repository architecture

```text
brisbane-crash-ml-lab/
├── .github/
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── configs/
│   ├── base.yaml
│   ├── smoke.yaml
│   ├── standard.yaml
│   └── extended.yaml
├── data/
│   ├── external/.gitkeep
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── samples/fixture.csv
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── data_provenance.md
│   ├── experiment_plan.md
│   ├── model_card.md
│   ├── release_checklist.md
│   └── prs/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   └── 03_error_analysis.ipynb
├── reports/
│   ├── figures/.gitkeep
│   ├── tables/.gitkeep
│   └── index.html
├── artifacts/
│   ├── models/.gitkeep
│   ├── metrics/.gitkeep
│   └── manifests/.gitkeep
├── src/crashlab/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── logging.py
│   ├── paths.py
│   ├── data/
│   │   ├── acquire.py
│   │   ├── schema.py
│   │   ├── validate.py
│   │   ├── clean.py
│   │   └── manifest.py
│   ├── features/
│   │   ├── targets.py
│   │   ├── temporal.py
│   │   ├── spatial.py
│   │   ├── encoders.py
│   │   └── build.py
│   ├── models/
│   │   ├── common.py
│   │   ├── binary.py
│   │   ├── multiclass.py
│   │   ├── ordinal.py
│   │   ├── anomalies.py
│   │   ├── hotspots.py
│   │   └── counts.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── calibration.py
│   │   ├── stability.py
│   │   ├── error_analysis.py
│   │   └── reports.py
│   └── app/
│       └── streamlit_app.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── .editorconfig
├── .gitignore
├── .pre-commit-config.yaml
├── CITATION.cff
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
└── PROJECT_OVERVIEW.md
```

Raw, interim, processed, and model binary files must not be committed. Small deterministic fixtures and lightweight example outputs may be committed.

---

## 9. Technology choices

### Core

- Python 3.11 or 3.12.
- `uv` for environment and lockfile management.
- `pandas` and `pyarrow` for tabular and Parquet workflows.
- `scikit-learn` for preprocessing, classification, anomaly detection, clustering, calibration, and metrics.
- `statsmodels` for Poisson and negative-binomial models.
- `scipy` for statistical checks and optional KDE.
- `requests` plus retry logic for acquisition.
- `pandera` or explicit typed validation for schema checks.
- `joblib` for model artifacts.
- `matplotlib` and `plotly` for reporting.
- `streamlit` for the local app.
- `typer` for CLI commands.
- `pytest`, `ruff`, and `mypy` for quality checks.

### Explicitly avoid by default

- GPU frameworks.
- Neural networks.
- Very large hyperparameter grids.
- Dense one-hot encoding of street names or crash reference IDs.
- Network calls during tests.
- Committing downloaded government data to Git.

---

## 10. CLI and Make targets

Expected command surface:

```bash
uv sync
uv run crashlab acquire --profile standard
uv run crashlab validate --profile standard
uv run crashlab prepare --profile standard
uv run crashlab train-binary --profile standard
uv run crashlab train-multiclass --profile standard
uv run crashlab train-ordinal --profile standard
uv run crashlab detect-anomalies --profile standard
uv run crashlab cluster-hotspots --profile standard
uv run crashlab train-counts --profile standard
uv run crashlab report --profile standard
uv run crashlab all --profile standard
uv run streamlit run src/crashlab/app/streamlit_app.py
```

Make targets:

```text
make setup
make lint
make typecheck
make test
make smoke
make download
make validate
make prepare
make train
make anomalies
make spatial
make counts
make report
make all
make app
make clean-generated
```

`make all` must execute the standard offline stages after acquisition. `make smoke` must run entirely on the committed fixture without internet access.

---

# 11. Delivery phases

## Phase 0 — Repository bootstrap and governance

### Build

- Initialise Git on `main`.
- Add Python project metadata, lockfile, source package, test structure, Makefile, and CI.
- Add `.gitignore` rules for datasets, caches, model binaries, notebook checkpoints, and local environment files.
- Add `LICENSE`, `CITATION.cff`, contribution notes, PR template, and issue templates.
- Add configuration profiles and central path management.
- Add structured logging and a run manifest format.

### Verification

- `uv sync` completes on CPU-only Python.
- `uv run crashlab --help` exits successfully.
- `make lint`, `make typecheck`, and `make test` pass.
- CI uses only the small fixture and performs no network download.
- A fresh clone can run `make smoke`.

### Success gate

The repository is installable, testable, and reproducible before any real data is downloaded.

---

## Phase 1 — Data acquisition

### Build

- Inspect OpenDataSoft dataset metadata and schema.
- Resolve canonical project columns dynamically.
- Query counts for Brisbane City and candidate year ranges.
- Request only selected rows and fields using the export API.
- Stream the response to a temporary file.
- Reject HTML/error responses masquerading as data.
- Verify content length while downloading and abort before 50 MiB.
- Atomically move a valid file into `data/raw`.
- Save SHA-256, byte size, row count, source URL, retrieval time, filters, selected fields, and remote metadata in a manifest.
- Preserve the raw file unchanged.

### Checks

- HTTP success and expected content type.
- File size greater than 100 KiB and no more than 50 MiB.
- At least 20,000 Brisbane City records where the source supports that range.
- No rows outside the configured LGA.
- Year range equals the accepted adaptive range.
- Required fields are present.
- Severity has expected values.
- Retrieval is idempotent unless `--force` is used.

### Failure handling

- Retry transient failures with exponential backoff.
- If the export exceeds 50 MiB, narrow the year range and retry.
- If it is too small, expand backwards to 2011.
- If the mirror is unavailable, retain a useful error report and run the smoke pipeline; do not download the full 200 MiB file unless the explicit environment opt-in is set.

### Success gate

A verified, immutable Brisbane-only raw file and provenance manifest exist locally without breaching the download cap.

---

## Phase 2 — Data validation and cleaning

### Build

- Normalise column names to snake case.
- Coerce year, hour, coordinates, speed limit, casualty counts, and unit counts.
- Standardise whitespace, null sentinels, and category spelling.
- Remove exact duplicate rows.
- Flag duplicate crash reference numbers instead of blindly deleting them.
- Remove property-damage-only records.
- Mark rows with invalid coordinates, impossible hours, or unsupported severity.
- Produce both:
  - a cleaned analysis dataset;
  - a rejected-row table with reason codes.
- Save cleaned output as compressed Parquet.

### Data-quality checks

- `crash_year` is within the acquired year interval.
- `crash_hour` is 0–23.
- Latitude and longitude parse as numeric.
- Broad Brisbane sanity bounds: latitude between -28.2 and -26.8; longitude between 152.5 and 153.5.
- LGA equals Brisbane City after normalisation.
- Numeric counts are nonnegative.
- Severity is one of the expected casualty categories.
- Missingness and cardinality reports exist for every column.
- Coordinate-valid rate should normally exceed 98%; lower rates require an explicit warning.
- Rejected rows are less than 5% unless the source changed; higher rates fail the standard profile.

### Success gate

The cleaned dataset passes the contract, the rejection process is auditable, and raw data remains untouched.

---

## Phase 3 — Data audit and exploratory analysis

### Build

Produce reproducible tables and figures for:

- Rows by year and severity.
- Severe-class prevalence by year.
- Missingness and category cardinality.
- Crashes by hour, weekday, month, suburb, speed limit, road feature, lighting, and surface condition.
- Fatal records by year, with warnings about small numbers.
- Coordinate coverage and spatial density.
- Distribution shift between train, validation, and test years.
- Cramér's V or mutual information for selected categorical relationships.
- Duplicate and unusual-value summaries.

The notebook is explanatory, but all material tables must be generated by importable source code so results are reproducible outside Jupyter.

### Verification

- Every plot has a title, units, period, and source note.
- No chart exposes crash reference IDs.
- Percentages have denominators.
- Small-category suppression is applied in public-facing tables where appropriate.
- EDA runs from the cleaned Parquet file, not the raw CSV.

### Success gate

The report identifies class imbalance, temporal drift, missingness, and high-cardinality risks before modelling begins.

---

## Phase 4 — Feature engineering and temporal splits

### Build

Derived features may include:

- Cyclic hour: sine and cosine.
- Cyclic month: sine and cosine.
- Weekend, night, dawn/dusk, peak-hour, school-commute proxy.
- Numeric speed-limit buckets.
- Vehicle involvement indicators.
- Vulnerable-road-user indicator.
- Coarse spatial cells using rounded coordinates or a deterministic grid.
- Historical suburb-level counts computed strictly from prior training data.
- Frequency encoding for high-cardinality categories.

Default split for 2015–2023 data:

```text
Train:      2015–2020
Validation: 2021–2022
Test:       2023
```

If the adaptive period differs, allocate approximately the earliest 65% of years to training, the next 20% to validation, and the latest complete 15% to testing while keeping whole years intact.

### Checks

- Splits are mutually exclusive.
- Test year is never used for fitting encoders, imputers, thresholds, or hyperparameters.
- Validation is used for threshold selection and bounded tuning only.
- Category encoders define unknown-value behavior.
- Feature names and transformations are serialised.
- Leakage denylist is absent.
- Crash reference numbers and raw street names are absent from model features.

### Success gate

A single scikit-learn-compatible pipeline transforms raw cleaned rows into features without fitting on validation or test data.

---

## Phase 5 — Binary classification baselines

### Trials

Run these models in order:

1. Dummy classifier using class prior.
2. Logistic regression with class weights.
3. Shallow decision tree.
4. Random forest.
5. Extra Trees.
6. HistGradientBoostingClassifier.

Run each on:

- Contextual/pre-crash feature set.
- Immediate post-incident triage feature set.

### Default budget

- Three fixed seeds for lightweight models.
- One seed for expensive standard models plus a stability rerun for the selected champion.
- At most 15 random hyperparameter candidates.
- Three-fold time-aware or blocked CV only on train/validation data.
- `n_jobs=-1` where safe.
- No exhaustive grid search.

### Metrics

- Class prevalence.
- PR-AUC.
- ROC-AUC.
- Balanced accuracy.
- Precision, recall, and F1.
- Confusion matrix.
- Brier score.
- Calibration curve.
- Recall captured in the top 5%, 10%, and 20% risk bands.
- Metrics by year and selected large suburbs.

### Success criteria

A valid model passes the phase if it:

- Beats the dummy PR-AUC by at least 15% relative, or provides a clearly documented reason it cannot.
- Achieves balanced accuracy above 0.55 on the held-out test year.
- Captures at least 20% of severe crashes in the highest-risk 10% of predictions, where sample size permits.
- Has no leakage-test failures.
- Produces calibrated probabilities no worse than the uncalibrated champion after the selected calibration method.

Threshold-dependent criteria are secondary to probability-ranking and calibration metrics.

### Success gate

A champion is selected based on a declared metric hierarchy, not test-set cherry-picking.

---

## Phase 6 — Multiclass and ordinal classification

### Trials

Multiclass:

- Dummy stratified baseline.
- Multinomial logistic regression.
- Random forest or Extra Trees.
- HistGradientBoostingClassifier.

Ordinal:

- Three cumulative logistic models.
- Three cumulative HistGradientBoosting models if runtime allows.
- Probability-monotonicity correction.

### Metrics

- Macro F1.
- Weighted F1.
- Per-class precision and recall.
- Confusion matrix.
- Mean absolute class error.
- Quadratic weighted kappa.
- Fatal-class metrics reported with explicit uncertainty and sample counts.

### Success criteria

- Macro F1 exceeds the stratified dummy by at least 10% relative.
- Ordinal mean absolute class error is lower than the best ordinary multiclass baseline, or the result is documented as a negative finding.
- No model selection is based solely on fatal-class accuracy because that class is rare.

### Success gate

The project can explain whether the ordering of severity adds measurable value.

---

## Phase 7 — Anomaly and outlier detection

### Trials

- Rule-based invalid-value detector.
- Robust univariate and grouped z-score detector.
- Isolation Forest.
- Local Outlier Factor on a bounded sample.
- Optional one-class SVM only on a small sample in the extended profile.

Use low-dimensional, leakage-free features. Avoid treating a huge sparse one-hot matrix as the only representation.

### Evaluation without labels

- Inspect top 50 records per method.
- Categorise findings as:
  - source/data-quality issue;
  - legitimate rare event;
  - rare category combination;
  - location/time outlier;
  - unexplained.
- Compare top-k overlap between seeds and methods.
- Test score stability under small feature perturbations.
- Separate invalid records from valid-but-unusual records.

### Success criteria

- Output is deterministic given a seed.
- Top-50 review table contains human-readable reason features.
- Top-100 Jaccard overlap across two Isolation Forest seeds is at least 0.25, or instability is explicitly documented.
- Obvious invalid data does not dominate the legitimate-anomaly list after validation.

### Success gate

The anomaly result is interpretable and does not pretend to be fraud detection or causal discovery.

---

## Phase 8 — Spatial hotspot analysis

### Trials

1. Deterministic grid counts.
2. DBSCAN with haversine distance or projected coordinates.
3. HDBSCAN if available in the installed scikit-learn version.
4. Optional KDE on a bounded sample.

Run separately for:

- All crashes.
- Severe crashes.
- Night crashes.
- Wet-surface crashes.
- Motorcycle-involved crashes.
- Bicycle- or pedestrian-involved crashes.

### Checks

- Coordinate validity is checked before clustering.
- Latitude/longitude are converted to radians for haversine DBSCAN.
- Cluster hyperparameters are stated in physical units.
- Cluster sizes and noise fraction are reported.
- Results are not labelled as dangerous-road rankings without exposure data.

### Success criteria

- At least one method produces nontrivial clusters and a nonzero noise set.
- Cluster assignments are stable under a modest parameter perturbation, measured with adjusted Rand index or cluster-overlap summaries.
- Severe-only and all-crash maps can be compared without revealing individual identities.

### Success gate

The spatial output is reproducible, interpretable, and explicitly exposure-limited.

---

## Phase 9 — Suburb-month count modelling

### Dataset

Aggregate by suburb, year, and month. Generate lag features using only earlier periods.

### Trials

1. Overall mean.
2. Same-month historical mean.
3. Previous-month count.
4. Poisson regression.
5. Negative-binomial regression.
6. HistGradientBoostingRegressor with Poisson loss where supported.

### Diagnostics

- Mean-versus-variance table.
- Overdispersion statistic.
- Residual plots.
- Zero-count frequency.
- Error by suburb volume decile.

### Metrics

- Mean absolute error.
- Root mean squared error.
- Poisson deviance.
- Mean absolute scaled error against the seasonal baseline.

### Success criteria

- At least one model reduces Poisson deviance by 5% relative to the strongest simple baseline, or the negative result is documented.
- Negative-binomial dispersion and interpretation are reported.
- Test periods remain strictly later than training periods.

### Success gate

The project demonstrates when count-specific statistical models are or are not useful.

---

## Phase 10 — Explainability, calibration, robustness, and error analysis

### Build

- Logistic coefficients with uncertainty-aware interpretation.
- Permutation importance on held-out data.
- Partial dependence for a small approved feature set.
- Calibration comparison: none, sigmoid, isotonic where sample size permits.
- False-positive and false-negative review tables.
- Performance by year, broad time period, road-user involvement, and sufficiently large suburbs.
- Feature ablation:
  - no location;
  - no post-incident fields;
  - no weather/surface context;
  - no high-cardinality categories.
- Stability over seeds and bootstrap confidence intervals for headline metrics.

### Checks

- SHAP is optional and limited to a sample; it is not required.
- Feature importance is described as predictive association, not causality.
- Small subgroup metrics are suppressed or flagged as unreliable.
- Test set is not repeatedly used for iterative tuning.

### Success gate

The final report explains not only the winning score but where, when, and why the model fails.

---

## Phase 11 — Reporting and local application

### Report

Generate a static HTML report containing:

- Data provenance and limitations.
- Data-quality results.
- EDA highlights.
- Experiment table.
- Binary, multiclass, ordinal, anomaly, spatial, and count results.
- Runtime and memory observations.
- Leakage demonstration.
- Error analysis.
- Model card and recommended use/non-use.

### Streamlit application

Pages:

1. Dataset overview.
2. Severity model comparison.
3. Calibration and threshold explorer.
4. Anomaly explorer.
5. Hotspot map.
6. Suburb-month count results.
7. Provenance and limitations.

The app must read precomputed artifacts. It must not retrain models during page load.

### Success gate

A user can understand the project and inspect outputs without opening a notebook.

---

## Phase 12 — Hardening, release, and handoff

### Build

- Finalise README quickstart and architecture docs.
- Add model card and data provenance record.
- Add release checklist and changelog.
- Run clean-clone verification.
- Create `v0.1.0` tag.
- Save final runtime manifest, environment details, Git SHA, data hash, config hash, and selected model metrics.

### Release checks

- No raw data or secrets are committed.
- All tests pass without internet.
- Standard pipeline succeeds from a clean environment after the one-time download.
- Smoke pipeline succeeds without download.
- All generated paths are documented.
- License attribution is present.
- Known limitations are prominent.

### Success gate

The repository is reproducible and portfolio-ready.

---

# 12. Experiment matrix

| ID | Task | Feature set | Model | Main comparison |
|---|---|---|---|---|
| B00 | Binary | Context | Dummy prior | Baseline |
| B01 | Binary | Context | Logistic, balanced | Linear baseline |
| B02 | Binary | Context | Decision tree | Nonlinear interpretable |
| B03 | Binary | Context | Random forest | Bagging |
| B04 | Binary | Context | Extra Trees | Randomised bagging |
| B05 | Binary | Context | HistGradientBoosting | Boosting |
| B11–B15 | Binary | Triage | Same families | Value of post-incident fields |
| B90 | Binary | Invalid leakage demo | Logistic/tree | Demonstrate leakage |
| M00 | Multiclass | Triage | Dummy | Baseline |
| M01 | Multiclass | Triage | Multinomial logistic | Linear multiclass |
| M02 | Multiclass | Triage | Extra Trees | Nonlinear multiclass |
| M03 | Multiclass | Triage | HistGradientBoosting | Boosting |
| O01 | Ordinal | Triage | Cumulative logistic | Ordinal theory |
| O02 | Ordinal | Triage | Cumulative HGB | Nonlinear ordinal |
| A00 | Anomaly | Validity fields | Rules | Data-quality baseline |
| A01 | Anomaly | Compact mixed features | Isolation Forest | Global anomaly |
| A02 | Anomaly | Scaled sample | LOF | Local anomaly |
| S00 | Spatial | Coordinates | Grid counts | Deterministic baseline |
| S01 | Spatial | Coordinates | DBSCAN | Density clusters |
| S02 | Spatial | Coordinates | HDBSCAN | Variable density |
| C00 | Count | Suburb-month | Seasonal mean | Baseline |
| C01 | Count | Suburb-month | Poisson GLM | Equidispersed count model |
| C02 | Count | Suburb-month | Negative binomial | Overdispersed count model |
| C03 | Count | Suburb-month | HGB Poisson | Nonlinear count model |

All experiments must write a machine-readable metrics record with experiment ID, Git SHA, data hash, config hash, feature set, fit time, score time, memory notes, seed, and metrics.

---

# 13. Runtime budget

Suggested standard-profile budgets:

| Stage | Budget |
|---|---:|
| Acquisition and verification | 2 minutes excluding slow internet variance |
| Cleaning and validation | 2 minutes |
| EDA tables/figures | 2 minutes |
| Feature build | 2 minutes |
| Binary models and bounded tuning | 7 minutes |
| Multiclass and ordinal | 4 minutes |
| Anomaly and spatial | 3 minutes |
| Count models | 2 minutes |
| Report generation | 1 minute |
| Total | Approximately 20–25 minutes |

Runtime controls:

- Cap tree counts in standard mode.
- Cap tuning candidates and folds.
- Sample LOF, KDE, permutation importance, and optional explanation methods.
- Cache cleaned data and transformed matrices.
- Skip completed stages when hashes match.
- Record actual timings and warn when a stage exceeds its budget.

---

# 14. Verification framework

## 14.1 Data verification

- Source and schema metadata saved.
- SHA-256 saved and checked.
- Download ≤ 50 MiB.
- Row count ≥ 20,000 where source availability permits.
- LGA is Brisbane City for every accepted row.
- No property-damage-only target rows.
- Required columns exist.
- Year interval is as configured.
- Coordinates pass broad bounds.
- Duplicate and rejected-row reports exist.
- Latest included year policy is documented.

## 14.2 Pipeline verification

- Clean clone setup works.
- All commands are noninteractive.
- Each phase is idempotent.
- Interrupted downloads leave no accepted partial file.
- Cached artifacts are invalidated by data/config/code hash changes.
- Logs include stage timing and peak-memory approximation where practical.

## 14.3 ML verification

- No denylisted leakage fields.
- Whole-year split.
- Preprocessing fitted only on training data.
- Dummy baseline included.
- Test set evaluated once by the selection workflow.
- Calibration trained without test leakage.
- Headline metrics include uncertainty or stability information.
- Metrics and model artifact metadata agree.

## 14.4 Software verification

- `ruff check .`
- `ruff format --check .`
- `mypy src`
- `pytest`
- CLI smoke tests.
- Determinism regression test on the fixture.
- No network in unit tests or CI.
- Dependency vulnerability scan if available.

## 14.5 Documentation verification

- README commands match actual commands.
- Data source and license attribution are present.
- Limitations include reporting bias, exposure absence, preliminary recent data, and noncausal interpretation.
- Generated reports state data period and retrieval date.

---

# 15. Git, commit, PR, and merge plan

## 15.1 Branch policy

Primary branch: `main`.

Planned branches:

1. `chore/bootstrap`
2. `feat/data-acquisition`
3. `feat/data-validation`
4. `feat/eda-features`
5. `feat/classification`
6. `feat/anomaly-spatial`
7. `feat/count-models`
8. `feat/reporting-release`

No generated data or binary model files belong in Git.

## 15.2 Conventional commit plan

Recommended commits:

1. `chore: bootstrap cpu-first python project`
2. `ci: add offline quality and smoke checks`
3. `feat(data): add filtered crash data acquisition`
4. `test(data): verify download limits and provenance manifests`
5. `feat(data): add schema validation and cleaning pipeline`
6. `feat(features): add temporal splits and leakage-safe features`
7. `feat(models): add binary classification baselines`
8. `feat(models): add multiclass and ordinal experiments`
9. `feat(models): add anomaly detection and hotspot clustering`
10. `feat(models): add suburb-month count models`
11. `feat(report): add evaluation report and streamlit explorer`
12. `docs: finalise model card provenance and release guide`
13. `chore(release): prepare v0.1.0`

Every commit must leave tests passing for the implemented scope.

## 15.3 Pull-request plan

### PR 1 — Bootstrap

Includes repository scaffold, environment, CI, fixture, config, CLI skeleton, and documentation structure.

Gate: setup, lint, typing, tests, and smoke command pass.

### PR 2 — Acquisition

Includes schema introspection, filtered export, adaptive year range, download cap, checksums, and provenance.

Gate: integration tests use mocked HTTP; real acquisition command verified locally.

### PR 3 — Validation and preparation

Includes cleaning, rejection reasons, Parquet output, data-quality report, and data contract tests.

Gate: fixture and real subset pass required checks.

### PR 4 — EDA and features

Includes reproducible EDA generators, feature sets, temporal splitting, preprocessing, and leakage tests.

Gate: transformed train/validation/test matrices have expected shapes and no overlap.

### PR 5 — Classification

Includes binary, multiclass, ordinal, calibration, bounded search, metrics, and artifact registry.

Gate: all baselines run within profile budgets and main metrics are persisted.

### PR 6 — Anomaly and spatial

Includes rule anomalies, Isolation Forest, LOF, grid counts, DBSCAN/HDBSCAN, stability checks, and maps.

Gate: deterministic output and coordinate-method tests pass.

### PR 7 — Count models

Includes aggregation, lag features, Poisson, negative binomial, nonlinear count baseline, and diagnostics.

Gate: no future leakage in lag features; baseline comparison generated.

### PR 8 — Reporting and release

Includes HTML report, Streamlit app, model card, README, final runtime validation, changelog, and release tag.

Gate: clean-clone run and release checklist pass.

## 15.4 Merge policy

Preferred on a hosted Git platform:

- Required CI checks.
- Squash merge each PR into `main`.
- Delete merged branches.
- Tag `v0.1.0` after the final clean-clone verification.

Autonomous local fallback when no remote or authentication exists:

- Create each planned branch.
- Implement and commit the scoped work.
- Write a PR summary to `docs/prs/NN-title.md`.
- Merge into `main` using `git merge --no-ff` after local checks pass.
- Preserve a complete local audit trail.
- Do not invent a remote URL or claim that hosted PRs were created.

## 15.5 PR checklist

- [ ] Scope matches the phase.
- [ ] Tests added or updated.
- [ ] No raw data or secrets committed.
- [ ] Runtime impact measured.
- [ ] Documentation updated.
- [ ] Data/provenance changes recorded.
- [ ] Leakage and temporal-split checks pass.
- [ ] Generated files are either intentionally committed or ignored.
- [ ] Rollback or failure behavior is documented.

---

# 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Mirror is stale or incomplete | Save remote metadata, enforce latest-year policy, compare schema with official dictionary, publish date limitations |
| API field names change | Inspect schema dynamically and map aliases through a tested canonical mapping layer |
| File exceeds 50 MiB | Stream with byte cap; narrow year range automatically |
| High-cardinality memory blow-up | Exclude street names, use minimum-frequency one-hot or frequency encoding, sparse matrices, bounded categories |
| Severe class imbalance | PR-AUC, class weighting, calibrated probabilities, risk-band recall, no accuracy-only decisions |
| Target leakage | Denylist tests, prediction-moment feature sets, explicit leakage demonstration isolated from leaderboard |
| Temporal drift | Whole-year validation and test splits, metrics by year, drift report |
| Spatial clusters reflect traffic volume | State exposure limitation; do not interpret counts as intrinsic risk |
| Recent data is preliminary | Exclude latest preliminary period by default and document source status |
| Runtime exceeds budget | Profiles, stage budgets, bounded search, sampling, caching, per-stage timing |
| No GitHub credentials | Complete local branch/commit/merge audit and write PR-ready summaries |

---

# 17. Definition of done

The project is complete when all of the following are true:

- A fresh environment can be created with one documented command.
- The filtered Brisbane dataset can be downloaded without exceeding 50 MiB.
- Raw data, metadata, and checksum are preserved locally and ignored by Git.
- Validation creates cleaned Parquet, rejection records, and a quality report.
- The standard pipeline completes in no more than 25 minutes on the target machine, or a measured machine-specific exception is documented with a passing `fast` profile.
- Binary, multiclass, ordinal, anomaly, spatial, and count tasks all run.
- Dummy and statistical baselines are present.
- No leakage checks fail.
- Time-based validation is used.
- Model and experiment artifacts contain data, config, environment, and Git identifiers.
- A static report and Streamlit explorer are generated from precomputed outputs.
- Tests, linting, and typing pass.
- CI is offline and fixture-based.
- Git history follows the branch/commit/PR plan.
- README, provenance, model card, and limitations are complete.
- The repository is tagged `v0.1.0` after release verification.

---

# 18. Minimum final deliverables

1. `README.md` with setup, commands, outputs, and screenshots or generated examples.
2. `PROJECT_OVERVIEW.md` containing this plan.
3. Filtered acquisition and provenance pipeline.
4. Cleaned Parquet and validation report generated locally.
5. Reproducible feature and temporal-split pipeline.
6. Classification leaderboard and calibrated champion.
7. Multiclass-versus-ordinal analysis.
8. Anomaly review table.
9. Spatial hotspot comparison.
10. Poisson-versus-negative-binomial count analysis.
11. Static HTML report.
12. Streamlit application.
13. Tests, CI, model card, release checklist, and Git tag.

---

# 19. Source notes

- Queensland Government official dataset and caveats: <https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads>
- Queensland Government resource, schema, license, and size: <https://www.data.qld.gov.au/dataset/crash-data-from-queensland-roads/resource/e88943c0-5968-4972-a15f-38e120d72ec0>
- Queensland OpenDataSoft mirror: <https://queensland.opendatasoft.com/explore/dataset/road-crash-locations-queensland/>
- OpenDataSoft/Huwise Explore API v2.1 reference: <https://help.opendatasoft.com/apis/ods-explore-v2/>

The repository must record the actual metadata observed at execution time rather than assuming the dates in this document remain current.
