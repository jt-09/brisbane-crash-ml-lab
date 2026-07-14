"""Stability helpers for top-k overlap and cluster agreement."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import adjusted_rand_score  # type: ignore[import-untyped]


def jaccard_similarity(a: Iterable[object], b: Iterable[object]) -> float:
    """Jaccard index between two sets (0 when both empty)."""
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k highest scores (ties broken by index order)."""
    if k <= 0 or len(scores) == 0:
        return np.array([], dtype=int)
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")
    return order[:k]


def top_k_jaccard(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """Jaccard overlap of top-k index sets from two score vectors."""
    return jaccard_similarity(top_k_indices(scores_a, k), top_k_indices(scores_b, k))


def cluster_stability_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index between two cluster label assignments."""
    if len(labels_a) != len(labels_b):
        msg = "Label vectors must have equal length"
        raise ValueError(msg)
    return float(adjusted_rand_score(labels_a, labels_b))
