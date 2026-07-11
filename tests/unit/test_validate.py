"""Raw validation contract tests."""

from __future__ import annotations

import pandas as pd

from crashlab.data.schema import BRISBANE_LGA
from crashlab.data.validate import validate_raw_frame


def _minimal_row(**overrides: object) -> dict[str, object]:
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
    }
    row.update(overrides)
    return row


def test_validate_accepts_ods_severity_variants() -> None:
    df = pd.DataFrame(
        [
            _minimal_row(crash_severity="Medical treatment"),
            _minimal_row(crash_ref_number="FX0002", crash_severity="Minor injury"),
            _minimal_row(crash_ref_number="FX0003", crash_severity="Hospitalization"),
        ]
    )
    report = validate_raw_frame(df, year_start=2015, year_end=2023)
    assert report["passed"] is True
    assert "unknown_severity" not in str(report["issues"])


def test_validate_rejects_unknown_severity() -> None:
    df = pd.DataFrame([_minimal_row(crash_severity="Bogus severity")])
    report = validate_raw_frame(df, year_start=2015, year_end=2023)
    assert report["passed"] is False
    assert any("unknown_severity" in issue for issue in report["issues"])
