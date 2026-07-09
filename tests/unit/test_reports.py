"""Unit tests for HTML report generation."""

from __future__ import annotations

from pathlib import Path

from crashlab.config import load_config
from crashlab.evaluation.reports import REPORT_SECTIONS, build_html_report, run_report
from crashlab.paths import ensure_dirs


def test_report_html_contains_key_sections(repo_root: Path | None = None) -> None:
    from crashlab.config import find_repo_root

    root = repo_root or find_repo_root()
    config = load_config("smoke", repo_root=root)
    paths = ensure_dirs(config)
    metrics = {
        "binary": {},
        "multiclass": {},
        "ordinal": {},
        "anomalies": None,
        "hotspots": None,
        "counts": None,
        "explanation": None,
        "eda": None,
        "data_quality": None,
        "run_all": None,
    }
    html = build_html_report(config, paths, metrics)
    for section in REPORT_SECTIONS:
        assert f'id="{section}"' in html
    assert "predictive associations" in html.lower() or "not causal" in html.lower()


def test_run_report_smoke(repo_root: Path | None = None) -> None:
    from crashlab.config import find_repo_root

    root = repo_root or find_repo_root()
    config = load_config("smoke", repo_root=root)
    paths = ensure_dirs(config)
    result = run_report(config, paths, force=True)
    assert result["status"] == "completed"
    index = paths.reports_dir / "index.html"
    assert index.is_file()
    content = index.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "model_card" in content or "Model Card" in content
