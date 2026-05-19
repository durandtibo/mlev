r"""Immutable containers that store computed metric values.

Result objects expose a consistent API to combine partial results and export
aggregates as dictionaries.

Example:
    ```pycon
    >>> from mlev.results import AccuracyResult
    >>> result = AccuracyResult(num_correct_predictions=8, num_predictions=10)
    >>> result.accuracy
    0.8
    >>> result.to_dict()
    {'accuracy': 0.8, 'num_correct_predictions': 8, 'num_predictions': 10}

    ```
"""

from __future__ import annotations

__all__ = ["AccuracyResult", "BaseResult", "BinaryConfusionMatrixResult"]

from mlev.results.base import BaseResult
from mlev.results.classification.accuracy import AccuracyResult
from mlev.results.classification.binary_confmat import BinaryConfusionMatrixResult
