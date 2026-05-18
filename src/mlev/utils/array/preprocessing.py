r"""Utilities to preprocess ``numpy.ndarray`` with missing values."""

from __future__ import annotations

__all__ = ["preprocess", "preprocess_pred"]


from typing import TYPE_CHECKING

import numpy as np

from mlev.utils.array.missing import multi_is_missing
from mlev.utils.array.shape import check_same_shape

if TYPE_CHECKING:
    from collections.abc import Sequence


def preprocess(arrays: Sequence[np.ndarray], drop_missing: bool = False) -> list[np.ndarray]:
    r"""Preprocess a sequence of arrays by optionally removing rows with
    missing values.

    A value is considered missing if it is ``NaN`` or ``None``.

    Args:
        arrays: The arrays to preprocess. All arrays must have the
            same shape.
        drop_missing: If ``True``, the rows where any array has a
            missing value are removed, otherwise they are kept.

    Returns:
        A list of preprocessed arrays with the same order as the input.

    Raises:
        ValueError: if ``arrays`` is empty.
        ValueError: if the arrays do not all have the same shape.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.utils.array import preprocess
        >>> arrays = [np.array([1, 0, 0, 1, 1, np.nan]), np.array([0, 1, 0, 1, np.nan, 1])]
        >>> preprocess(arrays)
        [array([ 1.,  0.,  0.,  1.,  1., nan]), array([ 0.,  1.,  0.,  1., nan,  1.])]
        >>> preprocess(arrays, drop_missing=True)
        [array([1., 0., 0., 1.]), array([0., 1., 0., 1.])]

        ```
    """
    if len(arrays) == 0:
        msg = "'arrays' cannot be empty"
        raise ValueError(msg)
    check_same_shape(arrays)
    if not drop_missing:
        return list(arrays)
    mask = np.logical_not(multi_is_missing(arrays))
    return [arr[mask] for arr in arrays]


def preprocess_pred(
    y_true: np.ndarray, y_pred: np.ndarray, drop_missing: bool = False
) -> tuple[np.ndarray, np.ndarray]:
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
        >>> import numpy as np
        >>> from mlev.utils.array import preprocess_pred
        >>> y_true = np.array([1, 0, 0, 1, 1, np.nan])
        >>> y_pred = np.array([0, 1, 0, 1, np.nan, 1])
        >>> preprocess_pred(y_true, y_pred)
        (array([ 1.,  0.,  0.,  1.,  1., nan]), array([ 0.,  1.,  0.,  1., nan,  1.]))
        >>> preprocess_pred(y_true, y_pred, drop_missing=True)
        (array([1., 0., 0., 1.]), array([0., 1., 0., 1.]))

        ```
    """
    check_same_shape([y_true, y_pred])
    if not drop_missing:
        return y_true, y_pred
    mask = np.logical_not(multi_is_missing([y_true, y_pred]))
    return y_true[mask], y_pred[mask]
