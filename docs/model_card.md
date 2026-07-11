# Model card

## Model summary

Binary severity classifiers predict whether a reported Brisbane injury crash is severe (fatal or hospitalised), using leakage-safe feature moments and time-based validation.

## Intended use

- Exploratory risk scoring and error analysis on historical Brisbane records.
- Comparing model families and calibration under documented splits.

## Out of scope

- Operational road-safety decisions or resource allocation.
- Causal claims about infrastructure, enforcement, or behaviour.
- Jurisdictions or time periods outside the configured extract.

## Training data

- Brisbane City injury crashes (PDO excluded), years per profile config (default 2015–2023).
- Splits: train years ≤ 2021, validation 2022, test 2023.

## Metrics

Reported on held-out test year(s): PR-AUC, Brier, balanced accuracy, recall at top risk percentiles, calibration curves. Subgroup metrics flagged when *n* &lt; 30.

## Limitations

- Reporting and survivorship bias in crash records.
- No exposure adjustment (traffic volume, population).
- Preliminary recent-year data per Queensland metadata.
- Class imbalance — fatal events especially sparse.

## Ethical note

Predictive patterns must not be interpreted as blame or causation for drivers, suburbs, or road authorities.

See `reports/index.html` for profile-specific champion metrics after running `crashlab report`.
