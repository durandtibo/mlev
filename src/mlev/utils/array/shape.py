r"""Utilities to validate array shapes and dimensions."""

from __future__ import annotations

__all__ = ["check_array_ndim"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def check_array_ndim(arr: np.ndarray, ndim: int, name: str = "input") -> None:
    r"""Validate that an array has the expected number of dimensions.

    Args:
        arr: The array to validate.
        ndim: The expected number of dimensions.
        name: The name of the input, used in error messages.
            Defaults to ``"input"``.

    Raises:
        ValueError: If ``arr.ndim`` is different from ``ndim``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import check_array_ndim
        >>> check_array_ndim(np.ones((2, 3)), ndim=2)

        ```
    """
    if arr.ndim != ndim:
        msg = f"{name}: expected {ndim}D array, got shape {arr.shape}"
        raise ValueError(msg)
