r"""Utilities to inspect ``polars.DataFrame`` with missing values."""

from __future__ import annotations

__all__ = ["contains_missing"]

from typing import TYPE_CHECKING

from mlev.utils.missing import MissingPolicy, check_missing_policy

if TYPE_CHECKING:
    import polars as pl


def contains_missing(
    frame: pl.DataFrame,
    missing_policy: MissingPolicy = "propagate",
    name: str = "input",
) -> bool:
    r"""Indicate if the given DataFrame contains at least one missing
    value.

    Missing values are represented by ``None``.
    NaNs are not considered to be missing data in Polars.

    Args:
        frame: The DataFrame to check.
        missing_policy: The missing policy. The valid values are ``'omit'``,
            ``'propagate'``, or ``'raise'``.
        name: An optional name to be more precise about the DataFrame when
            the exception is raised.

    Returns:
        ``True`` if the DataFrame contains at least one missing value,
        ``False`` otherwise.

    Raises:
        ValueError: if the DataFrame contains at least one missing value and
            ``missing_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.utils.frame import contains_missing
        >>> contains_missing(pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        False
        >>> contains_missing(pl.DataFrame({"x": [1, None, 3], "y": [4, 5, 6]}))
        True

        ```
    """
    check_missing_policy(missing_policy)
    if frame.is_empty():
        return False
    has_missing = frame.null_count().sum_horizontal().sum() > 0
    if has_missing and missing_policy == "raise":
        msg = f"{name} contains at least one missing value"
        raise ValueError(msg)
    return has_missing
