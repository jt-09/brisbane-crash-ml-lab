"""HTTP-mocked OpenDataSoft acquisition tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crashlab.config import load_config
from crashlab.data.acquire import (
    build_export_url,
    build_field_mapping,
    build_where_clause,
    discover_remote_fields,
    run_acquire,
)
from crashlab.paths import ensure_dirs


@pytest.fixture
def repo_root() -> Path:
    from crashlab.config import find_repo_root

    return find_repo_root()


def test_build_where_clause_year_filter() -> None:
    config = load_config("standard")
    clause = build_where_clause(config, year_start=2015, year_end=2023)
    assert "crash_year >= date'2015'" in clause
    assert "crash_year <= date'2023'" in clause
    assert "Brisbane City" in clause


def test_discover_remote_fields_from_metadata() -> None:
    metadata = {"fields": [{"name": "crash_ref_number"}, {"name": "crash_severity"}]}
    fields = discover_remote_fields(metadata)
    assert "crash_ref_number" in fields


def test_build_field_mapping_resolves_aliases() -> None:
    remote = ["count_casualty_medicallytreated", "crash_ref_number"]
    mapping = build_field_mapping(remote)
    assert mapping["count_casualty_medically_treated"] == "count_casualty_medicallytreated"
    assert mapping["crash_ref_number"] == "crash_ref_number"


@patch("crashlab.data.acquire.requests.get")
def test_network_acquire_mocked(
    mock_get: MagicMock, repo_root: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CRASHLAB_ALLOW_NETWORK", "1")
    header = (
        "crash_ref_number,crash_severity,crash_year,crash_month,crash_day_of_week,"
        "crash_hour,crash_longitude,crash_latitude,loc_suburb,loc_local_government_area,"
        "crash_roadway_feature,crash_traffic_control,crash_speed_limit,"
        "crash_road_surface_condition,crash_atmospheric_condition,crash_lighting_condition,"
        "crash_road_horiz_align,crash_road_vert_align"
    )
    rows = [
        f"R{i:05d},Minor Injury,2020,1,Monday,8,153.02,-27.47,City,Brisbane City,"
        "Intersection,Stop sign,60,Sealed - Dry,Clear,Daylight,Straight,Level"
        for i in range(2000)
    ]
    csv_body = (header + "\n" + "\n".join(rows) + "\n").encode()

    def _response(url: str, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/exports/csv" in url:
            resp.headers = {"Content-Type": "text/csv"}
            resp.iter_content = lambda chunk_size: [csv_body]
        else:
            resp.headers = {"Content-Type": "application/json"}
            field_names = header.split(",")
            payload = {"fields": [{"name": name} for name in field_names], "total_count": 25}
            resp.json = MagicMock(return_value=payload)
        return resp

    mock_get.side_effect = _response

    config = load_config("standard", repo_root=repo_root)
    config.raw["preseed_raw"] = {"path": str(tmp_path / "missing.csv"), "sha256": "0" * 64}
    config.data["min_rows"] = 20
    config.data["min_raw_bytes"] = 100
    paths = ensure_dirs(config)

    result = run_acquire(config, paths, force=True)
    assert result["status"] == "completed"
    assert result["row_count"] == 2000
    assert mock_get.called


def test_export_url_contains_select_and_where(repo_root: Path) -> None:
    config = load_config("standard", repo_root=repo_root)
    mapping = build_field_mapping(list(discover_remote_fields({"fields": []})))
    url = build_export_url(config, year_start=2015, year_end=2023, field_mapping=mapping)
    assert "where=" in url
    assert "select=" in url
