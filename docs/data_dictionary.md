# Data dictionary (canonical subset)

Source: Queensland *Crash data from Queensland roads* — Road crash locations (CC BY 4.0).

## Core fields used in modelling

| Column | Description |
|--------|-------------|
| `crash_ref_number` | Record identifier (excluded from features) |
| `crash_year`, `crash_month` | Temporal fields |
| `crash_day_of_week`, `crash_hour` | Time of crash |
| `crash_severity` | Target source: Fatal, Hospitalised, Medical, Minor, etc. |
| `loc_suburb`, `loc_post_code`, `loc_abs_statistical_area_2` | Location |
| `loc_latitude`, `loc_longitude` | GDA2020 coordinates |
| `crash_roadway_feature`, `crash_traffic_control` | Road context |
| `crash_speed_limit` | Speed limit (km/h) |
| `crash_road_surface_condition`, `crash_atmospheric_condition`, `crash_lighting_condition` | Environment |
| `crash_nature`, `crash_type` | Post-incident descriptors (triage moment only) |
| `count_unit_*` | Involved unit counts (triage moment only) |
| `count_casualty_*` | **Leakage denylist** — never used as predictors in valid models |

## Derived targets

- `severe_binary` — 1 if severity ∈ {Fatal, Hospitalised}.
- `severity_class` / `severity_ordinal` — multiclass/ordinal targets.

## Prediction moments

- **context** — pre/post-incident descriptors excluded.
- **triage** — adds crash nature, type, unit counts.
- **leakage_demo** — deliberately includes casualty fields for teaching only.
