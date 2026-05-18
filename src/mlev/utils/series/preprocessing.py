r"""Utilities to preprocess ``polars.Series`` with missing values."""

from __future__ import annotations

__all__ = ["preprocess_pred"]

from typing import TYPE_CHECKING

from mlev.utils.series.missing import multi_is_missing
from mlev.utils.series.shape import check_same_shape

if TYPE_CHECKING:
    import polars as pl


def preprocess_pred(
    y_true: pl.Series, y_pred: pl.Series, drop_missing: bool = False
) -> tuple[pl.Series, pl.Series]:
    r"""Preprocess ``y_true`` and ``y_pred`` arrays.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        drop_missing: If ``True``, the rows where any of ``y_true`` or
            ``y_pred`` is null are removed, otherwise they are kept.

    Returns:
        A tuple with the preprocessed ``y_true`` and ``y_pred``
            arrays.

    Raises:
        ValueError: if ``'y_true'`` and ``'y_pred'`` have different
            shapes.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.utils.series import preprocess_pred
        >>> y_true = pl.Series("y_true", [1, 0, 0, 1, 1, None])
        >>> y_pred = pl.Series("y_pred", [0, 1, 0, 1, None, 1])
        >>> preprocess_pred(y_true, y_pred)
        (shape: (6,)
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
        ])
        >>> preprocess_pred(y_true, y_pred, drop_missing=True)
        (shape: (4,)
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
        ])

        ```
    """
    check_same_shape([y_true, y_pred])
    if not drop_missing:
        return y_true, y_pred
    mask = multi_is_missing([y_true, y_pred]).not_()
    return y_true.filter(mask), y_pred.filter(mask)
