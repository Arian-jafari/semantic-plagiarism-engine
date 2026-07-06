"""
Jaccard similarity implementation.
"""

from __future__ import annotations


def jaccard_similarity(
    set_a: set[str],
    set_b: set[str]
) -> float:
    """
    Compute Jaccard similarity between two sets.

    Parameters
    ----------
    set_a : set[str]
    set_b : set[str]

    Returns
    -------
    float
        Jaccard similarity in [0, 1]
    """

    if not set_a and not set_b:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union