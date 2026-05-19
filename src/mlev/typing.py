r"""Shared typing aliases used by the public API.

Example:
    ```pycon
    >>> import numpy as np
    >>> from mlev.functional.array import accuracy
    >>> accuracy(y_true=np.array([1, 0, 1]), y_pred=[1, 1, 1])
    AccuracyResult(num_correct_predictions=2, num_predictions=3)

    ```
"""

from __future__ import annotations

__all__ = ["ArrayLike"]

from typing import Any

import numpy as np
import polars as pl

ArrayLike = np.ndarray | pl.Series | list[Any] | tuple[Any, ...]
