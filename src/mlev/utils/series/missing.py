r"""Utilities to inspect ``polars.Series`` with missing values."""

from __future__ import annotations

__all__ = ["contains_missing"]

import functools
from typing import TYPE_CHECKING

from mlev.utils.missing import check_missing_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


def contains_missing(x: pl.Series, missing_policy: str = "propagate") -> bool:
    r"""Indicate if the given series contains at least one missing value.

    Missing values are represented by ``None``.
    NaNs are not considered to be missing data in Polars.

    Args:
        x: The series to check.
        missing_policy: The missing policy. The valid values are ``'omit'``,
            ``'propagate'``, or ``'raise'``.

    Returns:
        ``True`` if the series contains at least one missing value.

    Raises:
        ValueError: if the series contains at least one missing value and
            ``missing_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.utils.series import contains_missing
        >>> contains_missing(pl.Series("col", [1, 2, 3]))
        False
        >>> contains_missing(pl.Series("col", [1, None, 3]))
        True

        ```
    """
    check_missing_policy(missing_policy)
    has_missing = x.null_count() > 0
    if has_missing and missing_policy == "raise":
        msg = f"{x.name} contains at least one missing value"
        raise ValueError(msg)
    return has_missing


def multi_is_null(series: Sequence[pl.Series], name: str = "is_null") -> pl.Series:
    r"""Test element-wise for null for all input series and return result
    as a boolean series.

    Args:
        series: The input series to test. All the series must have the
            same shape.
        name: The name of the output boolean series. Defaults to ``'is_null'``.

    Returns:
        A boolean series. ``True`` where any series is null,
        ``False`` otherwise.

    Raises:
        ValueError: if ``series`` is empty.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.utils.series import multi_is_null
        >>> mask = multi_is_null(
        ...     [pl.Series("x", [1, 0, 0, 1, None]), pl.Series("y", [1, None, 0, 1, 1])]
        ... )
        >>> mask
        shape: (5,)
        Series: 'is_null' [bool]
        [
           false
           true
           false
           false
           true
        ]

        ```
    """
    if len(series) == 0:
        msg = "'series' cannot be empty"
        raise ValueError(msg)
    return functools.reduce(lambda a, b: a | b, (s.is_null() for s in series)).alias(name)
