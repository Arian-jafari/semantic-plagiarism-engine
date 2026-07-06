"""
Evaluation utilities for pair classification experiments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Binary classification metrics.
    """

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float


def binary_metrics(labels: list[int], predictions: list[int]) -> ClassificationMetrics:
    """
    Compute precision, recall, F1, and accuracy for binary labels.
    """

    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    if not labels:
        return ClassificationMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    tp = fp = tn = fn = 0
    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 1 and prediction == 0:
            fn += 1
        else:
            raise ValueError("labels and predictions must contain only 0 or 1.")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)

    return ClassificationMetrics(tp, fp, tn, fn, precision, recall, f1, accuracy)
