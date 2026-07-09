"""Integration: smoke pipeline includes report stage."""

from __future__ import annotations

from pathlib import Path

from crashlab.config import load_config
from crashlab.paths import ensure_dirs
from crashlab.pipeline import PIPELINE_STAGES, run_all


def test_smoke_all_includes_report(repo_root: Path | None = None) -> None:
    from crashlab.config import find_repo_root

    root = repo_root or find_repo_root()
    assert "report" in PIPELINE_STAGES
    config = load_config("smoke", repo_root=root)
    paths = ensure_dirs(config)
    timings = run_all(config, force=True)
    assert "report_seconds" in timings
    index = paths.reports_dir / "index.html"
    assert index.is_file()
