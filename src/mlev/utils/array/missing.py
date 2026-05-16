r"""Utilities to inspect arrays with missing values.

There are two ways to represent missing values:
- np.nan when using the dtype float
- None when using the dtype object
"""

from __future__ import annotations

__all__ = ["contains_missing", "contains_none", "is_missing", "multi_is_missing"]

from typing import TYPE_CHECKING

import numpy as np

from mlev.utils.array.nan import contains_nan
from mlev.utils.missing import check_missing_policy

if TYPE_CHECKING:
    from collections.abc import Sequence


def contains_missing(
    arr: np.ndarray, missing_policy: str = "propagate", name: str = "input"
) -> bool:
    r"""Indicate if the given array contains at least one missing value.

    Missing values are represented by a ``np.nan`` or ``None``.

    Args:
        arr: The array to check.
        missing_policy: The missing policy. The valid values are ``'omit'``,
            ``'propagate'``, or ``'raise'``.
        name: An optional name to be more precise about the array when
            the exception is raised.

    Returns:
        ``True`` if the array contains at least one missing value.

    Raises:
        ValueError: if the array contains at least one missing value and
            ``missing_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import contains_missing
        >>> bool(contains_missing(np.array([1, 2, 3])))
        False
        >>> bool(contains_missing(np.array([1, 2, np.nan])))
        True

        ```
    """
    check_missing_policy(missing_policy)
    has_missing = contains_nan(arr)
    if not has_missing and arr.dtype == object:
        has_missing |= contains_none(arr)
    if has_missing and missing_policy == "raise":
        msg = f"{name} contains at least one missing value"
        raise ValueError(msg)
    return has_missing


def contains_none(arr: np.ndarray) -> bool:
    """Check if an object-dtype NumPy array contains any None values.

    Tries two strategies in order:

    1. ``None in arr``: uses NumPy's optimized __contains__, which is
       vectorized and avoids a Python loop. Fast, but may raise TypeError
       or ValueError if an element's __eq__ returns a non-scalar (e.g.
       a nested NumPy array or pandas Series).

    2. ``any(x is None for x in arr.flat)``: falls back to a Python
       generator using identity checks (``is None``) rather than equality,
       which is safe for any element type. Slower but robust.

    Args:
        arr: A NumPy array, typically of dtype=object.

    Returns:
        True if any element is None, False otherwise.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import contains_none
        >>> contains_none(np.array([1, 2, 3]))
        False
        >>> contains_none(np.array([1, 2, None]))
        True
        >>> contains_none(np.array(["a", None, "c"]))
        True

        ```
    """
    try:
        return None in arr
    except (TypeError, ValueError):
        return any(x is None for x in arr.flat)


def is_missing(arr: np.ndarray) -> np.ndarray:
    r"""Test element-wise for missing values and return result as a
    boolean array.

    A value is considered missing if it is ``NaN`` or ``None``.

    Args:
        arr: The input array to test.

    Returns:
        A boolean array. ``True`` where the value is missing (``NaN`` or
        ``None``), ``False`` otherwise.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import is_missing
        >>> is_missing(np.array([1.0, 0.0, float("nan"), 1.0]))
        array([False, False,  True, False])
        >>> is_missing(np.array([1.0, 0.0, float("nan"), 1.0, None], dtype=object))
        array([False, False,  True, False,  True])

        ```
    """
    if arr.dtype == object:
        return np.array(
            [x is None or (isinstance(x, float) and np.isnan(x)) for x in arr.flat],
            dtype=bool,
        ).reshape(arr.shape)
    try:
        return np.isnan(arr)
    except TypeError:
        # Non-numeric dtypes (e.g. datetime64, str) cannot be missing
        return np.zeros(arr.shape, dtype=bool)


def multi_is_missing(arrays: Sequence[np.ndarray]) -> np.ndarray:
    r"""Test element-wise for missing values for all input arrays and
    return result as a boolean array.

    A value is considered missing if it is ``NaN`` or ``None``.

    Args:
        arrays: The input arrays to test. All the arrays must have the
            same shape.

    Returns:
        A boolean array. ``True`` where any array has a missing value,
        ``False`` otherwise.

    Raises:
        ValueError: if ``arrays`` is empty.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import multi_is_missing
        >>> mask = multi_is_missing(
        ...     [np.array([1, 0, 0, 1, np.nan]), np.array([1, np.nan, 0, 1, 1])]
        ... )
        >>> mask
        array([False,  True, False, False,  True])
        >>> mask = multi_is_missing(
        ...     [np.array([1, None, 0], dtype=object), np.array([None, 2, 0], dtype=object)]
        ... )
        >>> mask
        array([ True,  True, False])

        ```
    """
    if len(arrays) == 0:
        msg = "'arrays' cannot be empty"
        raise ValueError(msg)
    return np.logical_or.reduce([is_missing(a) for a in arrays])
