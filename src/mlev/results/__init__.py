r"""Result objects that hold computed evaluation values."""

from __future__ import annotations

__all__ = ["AccuracyResult", "BaseResult"]

from mlev.results.base import BaseResult
from mlev.results.classification import AccuracyResult
