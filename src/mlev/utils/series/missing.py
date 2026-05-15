r"""Utilities to inspect ``polars.Series`` with missing values."""

from __future__ import annotations

__all__ = ["contains_missing"]


from typing import TYPE_CHECKING

from mlev.utils.missing import check_missing_policy

if TYPE_CHECKING:
    import polars as pl


def contains_missing(x: pl.Series, missing_policy: str = "propagate") -> bool:
    r"""Indicate if the given series contains at least one missing value.

    Missing values are represented by ``None``.

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
