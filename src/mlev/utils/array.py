r"""Contain array-like utilities."""

from __future__ import annotations

__all__ = ["to_numpy", "to_numpy_1d"]

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from mlev.typing import ArrayLike


def to_numpy(x: ArrayLike, name: str = "input") -> np.ndarray:
    r"""Convert an array-like object to a NumPy array.

    Supported input types are :class:`numpy.ndarray`,
    :class:`polars.Series`, :class:`list`, and :class:`tuple`.

    Args:
        x: The array-like object to convert.
        name: The name of the input, used in error messages.
            Defaults to ``"input"``.

    Returns:
        A NumPy array representation of ``x``. If ``x`` is already a
            :class:`numpy.ndarray`, it is returned as-is without copying.

    Raises:
        TypeError: If ``x`` is not a supported array-like type.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import to_numpy
        >>> to_numpy([1, 2, 3])
        array([1, 2, 3])
        >>> to_numpy((1, 2, 3))
        array([1, 2, 3])

        ```
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pl.Series):
        return x.to_numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    msg = f"{name}: unsupported type {type(x)}"
    raise TypeError(msg)


def to_numpy_1d(x: ArrayLike, name: str = "input") -> np.ndarray:
    r"""Convert an array-like object to a 1D NumPy array.

    Supported input types are the same as :func:`to_numpy`.

    Args:
        x: The array-like object to convert. Must be 1-dimensional
            after conversion.
        name: The name of the input, used in error messages.
            Defaults to ``"input"``.

    Returns:
        A 1D NumPy array representation of ``x``.

    Raises:
        TypeError: If ``x`` is not a supported array-like type.
        ValueError: If ``x`` does not have exactly one dimension after
            conversion.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import to_numpy_1d
        >>> to_numpy_1d([1, 2, 3])
        array([1, 2, 3])

        ```
    """
    arr = to_numpy(x, name=name)
    if arr.ndim != 1:
        msg = f"{name}: expected 1D array, got shape {arr.shape}"
        raise ValueError(msg)
    return arr
