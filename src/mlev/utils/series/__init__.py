r"""Helpers to validate and convert series-like inputs."""

from __future__ import annotations

__all__ = ["check_same_shape", "series_contains_missing"]

from mlev.utils.series.missing import series_contains_missing
from mlev.utils.series.shape import check_same_shape
