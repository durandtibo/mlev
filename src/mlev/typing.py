r"""Contain typing utilities."""

from __future__ import annotations

__all__ = ["ArrayLike"]


import numpy as np
import polars as pl

ArrayLike = np.ndarray | pl.Series | list | tuple
