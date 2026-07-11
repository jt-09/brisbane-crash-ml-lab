"""Cleaning and rejection logic tests."""

from __future__ import annotations

import pandas as pd

from crashlab.data.clean import assign_rejection_reasons, split_clean_rejected
from crashlab.data.schema import BRISBANE_LGA, SEVERITY_PDO


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "crash_ref_number": "FX0001",
        "crash_severity": "Minor Injury",
        "crash_year": 2020,
        "crash_month": 6,
        "crash_day_of_week": "Monday",
        "crash_hour": 10,
        "crash_longitude": 153.02,
        "crash_latitude": -27.47,
        "loc_suburb": "Brisbane City",
        "loc_local_government_area": BRISBANE_LGA,
        "crash_roadway_feature": "Intersection",
        "crash_traffic_control": "Stop sign",
        "crash_speed_limit": 60,
        "crash_road_surface_condition": "Sealed - Dry",
        "crash_atmospheric_condition": "Clear",
        "crash_lighting_condition": "Daylight",
        "crash_road_horiz_align": "Straight",
        "crash_road_vert_align": "Level",
        "count_casualty_minor_injury": 1,
    }
    row.update(overrides)
    return row


def test_pdo_rows_rejected_with_reason() -> None:
    df = pd.DataFrame([_base_row(crash_severity=SEVERITY_PDO)])
    reasons = assign_rejection_reasons(df, year_start=2015, year_end=2023)
    assert reasons.iloc[0] == "PDO_EXCLUDED"


def test_invalid_hour_rejected() -> None:
    df = pd.DataFrame([_base_row(crash_hour=25)])
    reasons = assign_rejection_reasons(df, year_start=2015, year_end=2023)
    assert "INVALID_HOUR" in reasons.iloc[0]


def test_invalid_coordinates_rejected() -> None:
    df = pd.DataFrame([_base_row(crash_latitude=0.0, crash_longitude=0.0)])
    reasons = assign_rejection_reasons(df, year_start=2015, year_end=2023)
    assert "INVALID_COORDINATES" in reasons.iloc[0]


def test_exact_duplicate_rows_rejected() -> None:
    rows = [_base_row(), _base_row()]
    df = pd.DataFrame(rows)
    clean, rejected = split_clean_rejected(df, year_start=2015, year_end=2023)
    assert len(clean) == 1
    assert len(rejected) == 1
    assert "DUPLICATE_ROW" in rejected["rejection_reason"].iloc[0]


def test_alias_columns_normalised_on_clean() -> None:
    row = _base_row()
    row["count_casualty_medicallytreated"] = 0
    del row["count_casualty_minor_injury"]
    df = pd.DataFrame([row])
    clean, _rejected = split_clean_rejected(df, year_start=2015, year_end=2023)
    assert "count_casualty_medically_treated" in clean.columns


def test_severity_mapping_retained_for_modelling_set() -> None:
    df = pd.DataFrame(
        [
            _base_row(crash_severity="Fatal"),
            _base_row(crash_ref_number="FX0002", crash_severity="Medical Treatment"),
        ]
    )
    clean, rejected = split_clean_rejected(df, year_start=2015, year_end=2023)
    assert len(clean) == 2
    assert len(rejected) == 0
    assert set(clean["crash_severity"]) == {"Fatal", "Medical Treatment"}


def test_ods_severity_variants_normalised_on_clean() -> None:
    df = pd.DataFrame(
        [
            _base_row(crash_severity="Medical treatment"),
            _base_row(crash_ref_number="FX0002", crash_severity="Minor injury"),
            _base_row(crash_ref_number="FX0003", crash_severity="Hospitalization"),
        ]
    )
    clean, rejected = split_clean_rejected(df, year_start=2015, year_end=2023)
    assert len(clean) == 3
    assert len(rejected) == 0
    assert set(clean["crash_severity"]) == {
        "Medical Treatment",
        "Minor Injury",
        "Hospitalisation",
    }
