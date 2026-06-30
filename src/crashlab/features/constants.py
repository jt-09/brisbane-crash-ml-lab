"""Feature-set constants, leakage policy, and column groupings."""

from __future__ import annotations

from typing import Final

# Absolute leakage denylist — never predictors in valid severity pipelines.
LEAKAGE_DENYLIST: Final[frozenset[str]] = frozenset(
    {
        "crash_severity",
        "count_casualty_fatality",
        "count_casualty_hospitalised",
        "count_casualty_medically_treated",
        "count_casualty_minor_injury",
        "count_casualty_total",
        "severe_binary",
        "severity_class",
        "severity_ordinal",
    }
)

LEAKAGE_DENYLIST_PATTERNS: Final[tuple[str, ...]] = (
    "casualty",
    "fatality",
    "hospitalised",
    "medically_treated",
    "minor_injury",
)

# Identifiers and high-risk text fields excluded from all model matrices.
IDENTIFIER_DENYLIST: Final[frozenset[str]] = frozenset(
    {
        "crash_ref_number",
        "duplicate_ref_flag",
        "rejection_reason",
    }
)

CONTEXT_EXCLUDED: Final[frozenset[str]] = frozenset(
    {
        "crash_nature",
        "crash_type",
        "crash_dca_code",
        "crash_dca_group_description",
        "dca_key_approach_dir",
        "count_unit_car",
        "count_unit_motorcycle_moped",
        "count_unit_truck",
        "count_unit_bus",
        "count_unit_bicycle",
        "count_unit_pedestrian",
        "count_unit_other",
    }
)

TRIAGE_EXTRA_CATEGORICAL: Final[tuple[str, ...]] = (
    "crash_nature",
    "crash_type",
    "crash_dca_group_description",
    "dca_key_approach_dir",
)

TRIAGE_EXTRA_NUMERIC: Final[tuple[str, ...]] = (
    "count_unit_car",
    "count_unit_motorcycle_moped",
    "count_unit_truck",
    "count_unit_bus",
    "count_unit_bicycle",
    "count_unit_pedestrian",
    "count_unit_other",
)

CONTEXT_CATEGORICAL: Final[tuple[str, ...]] = (
    "loc_suburb",
    "loc_post_code",
    "loc_abs_statistical_area_2",
    "crash_controlling_authority",
    "crash_roadway_feature",
    "crash_traffic_control",
    "crash_road_surface_condition",
    "crash_atmospheric_condition",
    "crash_lighting_condition",
    "crash_road_horiz_align",
    "crash_road_vert_align",
    "crash_day_of_week",
    "speed_bucket",
    "spatial_cell",
)

LEAKAGE_DEMO_EXTRA: Final[tuple[str, ...]] = (
    "count_casualty_fatality",
    "count_casualty_hospitalised",
    "count_casualty_medically_treated",
    "count_casualty_minor_injury",
    "count_casualty_total",
)

DERIVED_NUMERIC: Final[tuple[str, ...]] = (
    "crash_hour_sin",
    "crash_hour_cos",
    "crash_month_sin",
    "crash_month_cos",
    "is_weekend",
    "is_night",
    "is_dawn_dusk",
    "is_peak_hour",
    "is_school_commute",
    "is_vru_context",
    "crash_speed_limit",
)

FEATURE_MOMENTS: Final[tuple[str, ...]] = ("context", "triage", "leakage_demo")
