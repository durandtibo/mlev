r"""Utilities to preprocess ``numpy.ndarray`` with missing values."""

from __future__ import annotations

__all__ = ["preprocess_pred"]


import numpy as np

from mlev.utils.array.missing import multi_is_missing
from mlev.utils.array.shape import check_same_shape


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

    Examples:
        >>> import numpy as np
        >>> from mlev.utils.array import preprocess_pred
        >>> y_true = np.array([1, 0, 0, 1, 1, np.nan])
        >>> y_pred = np.array([0, 1, 0, 1, np.nan, 1])
        >>> preprocess_pred(y_true, y_pred)
        (array([ 1.,  0.,  0.,  1.,  1., nan]), array([ 0.,  1.,  0.,  1., nan,  1.]))
        >>> preprocess_pred(y_true, y_pred, drop_missing=True)
        (array([1., 0., 0., 1.]), array([0., 1., 0., 1.]))
    """
    check_same_shape([y_true, y_pred])
    if not drop_missing:
        return y_true, y_pred
    mask = np.logical_not(multi_is_missing([y_true, y_pred]))
    return y_true[mask], y_pred[mask]
