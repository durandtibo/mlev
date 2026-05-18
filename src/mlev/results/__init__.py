r"""Result objects that hold computed evaluation values."""

from __future__ import annotations

__all__ = ["AccuracyResult", "BaseResult", "BinaryConfusionMatrixResult"]

from mlev.results.base import BaseResult
from mlev.results.classification.accuracy import AccuracyResult
from mlev.results.classification.binary_confmat import BinaryConfusionMatrixResult
