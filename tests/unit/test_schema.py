"""Tests for canonical schema definitions."""

from __future__ import annotations

from crashlab.data.schema import (
    BRISBANE_LGA,
    FIELD_ALIASES,
    PRESEED_SHA256,
    REQUIRED_FIELDS,
    SEVERITY_PDO,
    SEVERITY_VALUES,
    normalize_column_name,
    normalize_severity,
)


def test_brisbane_lga_constant() -> None:
    assert BRISBANE_LGA == "Brisbane City"


def test_preseed_hash_uppercase() -> None:
    assert PRESEED_SHA256.upper() == PRESEED_SHA256
    assert len(PRESEED_SHA256) == 64


def test_severity_values_exclude_pdo() -> None:
    assert SEVERITY_PDO not in SEVERITY_VALUES
    assert "Fatal" in SEVERITY_VALUES
    assert "Hospitalisation" in SEVERITY_VALUES


def test_field_aliases_map_ods_quirks() -> None:
    assert FIELD_ALIASES["count_casualty_medicallytreated"] == "count_casualty_medically_treated"
    assert FIELD_ALIASES["count_casualty_minorinjury"] == "count_casualty_minor_injury"


def test_normalize_column_name_applies_aliases() -> None:
    assert normalize_column_name("Count_Casualty_MedicallyTreated") == (
        "count_casualty_medically_treated"
    )
    assert normalize_column_name("crash_ref_number") == "crash_ref_number"


def test_normalize_severity_aliases() -> None:
    assert normalize_severity("fatal") == "Fatal"
    assert normalize_severity("pdo") == "Property Damage Only"
    assert normalize_severity("Hospitalisation") == "Hospitalisation"


def test_required_fields_subset_of_canonical_usage() -> None:
    assert "crash_ref_number" in REQUIRED_FIELDS
    assert "crash_severity" in REQUIRED_FIELDS
    assert "loc_local_government_area" in REQUIRED_FIELDS
