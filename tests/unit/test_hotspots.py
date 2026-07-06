"""Spatial clustering smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crashlab.models.hotspots import dbscan_cluster, grid_count_summary, valid_coordinates


def _coord_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crash_longitude": [153.02, 153.021, 153.05, 153.051, 153.08],
            "crash_latitude": [-27.47, -27.471, -27.50, -27.501, -27.53],
            "spatial_cell": ["a", "a", "b", "b", "c"],
        }
    )


def test_valid_coordinates_filters_bbox() -> None:
    df = pd.DataFrame(
        {
            "crash_longitude": [153.0, 200.0],
            "crash_latitude": [-27.5, -27.5],
        }
    )
    valid = valid_coordinates(df)
    assert len(valid) == 1


def test_grid_count_summary_deterministic() -> None:
    df = _coord_frame()
    a = grid_count_summary(df)
    b = grid_count_summary(df)
    assert a["n_cells"] == b["n_cells"]
    assert a["top_cells"] == b["top_cells"]


def test_dbscan_haversine_produces_clusters_or_noise() -> None:
    df = _coord_frame()
    result = dbscan_cluster(df, eps_meters=800.0, min_samples=2)
    assert result["n_points"] == len(df)
    assert "noise_fraction" in result
    labels = np.asarray(result["labels"])
    assert len(labels) == len(df)
