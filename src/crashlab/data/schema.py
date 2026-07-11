"""Canonical data contract, field aliases, and validation constants."""

from __future__ import annotations

from typing import Final

BRISBANE_LGA: Final[str] = "Brisbane City"

PRESEED_SHA256: Final[str] = "CE2A0435366E0870F238295CB9A2A700C12F6FFB25815B8D51ACA2A506CEFF22"

# Casualty severities used in modelling (PDO excluded).
SEVERITY_VALUES: Final[tuple[str, ...]] = (
    "Fatal",
    "Hospitalisation",
    "Medical Treatment",
    "Minor Injury",
)

SEVERITY_PDO: Final[str] = "Property Damage Only"

ALL_SEVERITY_VALUES: Final[tuple[str, ...]] = SEVERITY_VALUES + (SEVERITY_PDO,)

SEVERITY_ALIASES: Final[dict[str, str]] = {
    "fatal": "Fatal",
    "hospitalisation": "Hospitalisation",
    "hospitalization": "Hospitalisation",
    "hospitalized": "Hospitalisation",
    "hospitalised": "Hospitalisation",
    "medical treatment": "Medical Treatment",
    "minor injury": "Minor Injury",
    "property damage only": "Property Damage Only",
    "pdo": "Property Damage Only",
}

CANONICAL_FIELDS: Final[tuple[str, ...]] = (
    "crash_ref_number",
    "crash_severity",
    "crash_year",
    "crash_month",
    "crash_day_of_week",
    "crash_hour",
    "crash_nature",
    "crash_type",
    "crash_longitude",
    "crash_latitude",
    "loc_suburb",
    "loc_local_government_area",
    "loc_post_code",
    "loc_abs_statistical_area_2",
    "crash_controlling_authority",
    "crash_roadway_feature",
    "crash_traffic_control",
    "crash_speed_limit",
    "crash_road_surface_condition",
    "crash_atmospheric_condition",
    "crash_lighting_condition",
    "crash_road_horiz_align",
    "crash_road_vert_align",
    "crash_dca_code",
    "crash_dca_group_description",
    "dca_key_approach_dir",
    "count_casualty_fatality",
    "count_casualty_hospitalised",
    "count_casualty_medically_treated",
    "count_casualty_minor_injury",
    "count_casualty_total",
    "count_unit_car",
    "count_unit_motorcycle_moped",
    "count_unit_truck",
    "count_unit_bus",
    "count_unit_bicycle",
    "count_unit_pedestrian",
    "count_unit_other",
)

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "crash_ref_number",
    "crash_severity",
    "crash_year",
    "crash_month",
    "crash_day_of_week",
    "crash_hour",
    "crash_longitude",
    "crash_latitude",
    "loc_suburb",
    "loc_local_government_area",
    "crash_roadway_feature",
    "crash_traffic_control",
    "crash_speed_limit",
    "crash_road_surface_condition",
    "crash_atmospheric_condition",
    "crash_lighting_condition",
    "crash_road_horiz_align",
    "crash_road_vert_align",
)

# OpenDataSoft / source quirks → canonical snake_case names.
FIELD_ALIASES: Final[dict[str, str]] = {
    "count_casualty_medicallytreated": "count_casualty_medically_treated",
    "count_casualty_minorinjury": "count_casualty_minor_injury",
    "crash_ref_no": "crash_ref_number",
    "crash_ref": "crash_ref_number",
    "loc_lga": "loc_local_government_area",
    "loc_localgovernmentarea": "loc_local_government_area",
}

# Broad Brisbane / south-east QLD sanity bounds (do not over-filter).
LAT_MIN: Final[float] = -28.2
LAT_MAX: Final[float] = -26.8
LON_MIN: Final[float] = 152.5
LON_MAX: Final[float] = 153.5

HOUR_MIN: Final[int] = 0
HOUR_MAX: Final[int] = 23

NULL_SENTINELS: Final[frozenset[str]] = frozenset(
    {"", "na", "n/a", "null", "none", ".", "-", "unknown", "nan"}
)

MONTH_NAME_TO_INT: Final[dict[str, int]] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

COUNT_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in CANONICAL_FIELDS if name.startswith("count_")
)

REJECTION_CODES: Final[tuple[str, ...]] = (
    "INVALID_YEAR",
    "INVALID_HOUR",
    "INVALID_SEVERITY",
    "INVALID_LGA",
    "INVALID_COORDINATES",
    "INVALID_COUNT",
    "DUPLICATE_ROW",
    "PDO_EXCLUDED",
    "MISSING_REQUIRED",
)


def normalize_column_name(name: str) -> str:
    """Map a raw column name to its canonical snake_case identifier."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return FIELD_ALIASES.get(key, key)


def normalize_severity(value: object) -> str | None:
    """Standardise severity strings; return None when unrecognised."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in ALL_SEVERITY_VALUES:
        return text
    lowered = text.lower()
    if lowered in SEVERITY_ALIASES:
        return SEVERITY_ALIASES[lowered]
    for canonical in ALL_SEVERITY_VALUES:
        if canonical.lower() == lowered:
            return canonical
    return None
