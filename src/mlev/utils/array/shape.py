r"""Contain array-like utilities."""

from __future__ import annotations

__all__ = ["check_array_ndim"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def check_array_ndim(arr: np.ndarray, ndim: int) -> None:
    r"""Check if the number of array dimensions is matching the target
    number of dimensions.

    Args:
        arr: The array to check.
        ndim: The targeted number of array dimensions.

    Raises:
        ValueError: if the number of array dimensions does not match.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from mlev.utils.array import check_array_ndim
    >>> check_array_ndim(np.ones((2, 3)), ndim=2)

    ```
    """
    if arr.ndim != ndim:
        msg = f"Incorrect number of array dimensions: {arr.ndim} (expected: {ndim})"
        raise ValueError(msg)
