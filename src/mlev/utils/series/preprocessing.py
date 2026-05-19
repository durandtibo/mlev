r"""Utilities to preprocess ``polars.Series`` with missing values."""

from __future__ import annotations

__all__ = ["preprocess"]

from typing import TYPE_CHECKING

from mlev.utils.series.missing import multi_is_missing
from mlev.utils.series.shape import check_same_shape

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


def preprocess(series: Sequence[pl.Series], drop_missing: bool = False) -> list[pl.Series]:
    r"""Preprocess a sequence of series by optionally removing rows with
    missing values.

    Missing values are represented by ``None``.
    NaNs are not considered to be missing data in Polars.

    Args:
        series: The series to preprocess. All series must have the
            same shape.
        drop_missing: If ``True``, the rows where any series has a
            missing value are removed, otherwise they are kept.

    Returns:
        A list of preprocessed series with the same length and order
        as the input. Returns an empty list when ``series`` is empty.

    Raises:
        ValueError: if the series do not all have the same shape.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.utils.series import preprocess
        >>> series = [
        ...     pl.Series("y_true", [1, 0, 0, 1, 1, None]),
        ...     pl.Series("y_pred", [0, 1, 0, 1, None, 1]),
        ... ]
        >>> preprocess(series)
        [shape: (6,)
        Series: 'y_true' [i64]
        [
            1
            0
            0
            1
            1
            null
        ], shape: (6,)
        Series: 'y_pred' [i64]
        [
            0
            1
            0
            1
            null
            1
        ]]
        >>> preprocess(series, drop_missing=True)
        [shape: (4,)
        Series: 'y_true' [i64]
        [
            1
            0
            0
            1
        ], shape: (4,)
        Series: 'y_pred' [i64]
        [
            0
            1
            0
            1
        ]]

        ```
    """
    if not series:
        return []
    check_same_shape(series)
    if not drop_missing:
        return list(series)
    mask = multi_is_missing(series).not_()
    return [s.filter(mask) for s in series]
