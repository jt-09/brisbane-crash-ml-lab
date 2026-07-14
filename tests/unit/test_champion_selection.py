"""Champion selection and leakage_demo exclusion tests."""

from __future__ import annotations

from crashlab.models.common import (
    LEADERBOARD_EXCLUDED_MOMENTS,
    select_champion,
    summarize_leaderboard,
)


def test_leakage_demo_excluded_from_champion() -> None:
    candidates = [
        {
            "model_name": "logistic",
            "moment": "leakage_demo",
            "valid": True,
            "val_metrics": {"pr_auc": 0.99, "brier": 0.01, "recall_at_top_10pct": 0.9},
            "fit_seconds": 1.0,
        }
    ]
    assert select_champion(candidates, moment="leakage_demo") is None
    assert "leakage_demo" in LEADERBOARD_EXCLUDED_MOMENTS


def test_champion_picks_highest_val_pr_auc() -> None:
    candidates = [
        {
            "model_name": "dummy",
            "moment": "context",
            "valid": True,
            "val_metrics": {"pr_auc": 0.40, "brier": 0.20, "recall_at_top_10pct": 0.1},
            "fit_seconds": 0.1,
        },
        {
            "model_name": "logistic",
            "moment": "context",
            "valid": True,
            "val_metrics": {"pr_auc": 0.55, "brier": 0.18, "recall_at_top_10pct": 0.2},
            "fit_seconds": 0.5,
        },
    ]
    champion = select_champion(candidates, moment="context", baseline_pr_auc=0.40)
    assert champion is not None
    assert champion["model_name"] == "logistic"
    assert champion.get("is_champion") is True


def test_leaderboard_excludes_leakage_demo_moment() -> None:
    candidates = [
        {
            "model_name": "logistic",
            "moment": "leakage_demo",
            "valid": True,
            "val_metrics": {"pr_auc": 0.99, "brier": 0.01},
            "fit_seconds": 1.0,
        },
        {
            "model_name": "logistic",
            "moment": "context",
            "valid": True,
            "val_metrics": {"pr_auc": 0.50, "brier": 0.15},
            "fit_seconds": 1.0,
        },
    ]
    board = summarize_leaderboard(candidates, moment="context")
    assert len(board) == 1
    assert board[0]["model_name"] == "logistic"
    assert summarize_leaderboard(candidates, moment="leakage_demo") == []
