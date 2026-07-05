"""Stability helper unit tests."""

from __future__ import annotations

import numpy as np

from crashlab.evaluation.stability import jaccard_similarity, top_k_jaccard


def test_jaccard_identical_sets() -> None:
    assert jaccard_similarity([1, 2, 3], [1, 2, 3]) == 1.0


def test_top_k_jaccard_partial_overlap() -> None:
    scores_a = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
    scores_b = np.array([1.0, 10.0, 9.0, 8.0, 0.0])
    j = top_k_jaccard(scores_a, scores_b, k=3)
    assert j == 0.5
