r"""Contain code to compute the accuracy metric."""

from __future__ import annotations

__all__ = ["accuracy"]

from typing import TYPE_CHECKING

import polars as pl

from mlev.results import AccuracyResult
from mlev.utils import array, series
from mlev.utils.array import to_numpy_1d

if TYPE_CHECKING:
    import numpy as np

    from mlev.typing import ArrayLike
    from mlev.utils.missing import MissingPolicy


def accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    missing_policy: MissingPolicy = "propagate",
) -> AccuracyResult:
    r"""Compute the accuracy score.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``.

    Returns:
        The accuracy result.

    Raises:
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` contains missing values and
            ``missing_policy`` is ``'raise'``.
        ValueError: if ``y_pred`` contains missing values and
            ``missing_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import polars as pl
        >>> from mlev.functional.array import accuracy
        >>> # with numpy arrays
        >>> accuracy(
        ...     y_true=np.array([1, 0, 0, 1, 1]),
        ...     y_pred=np.array([1, 0, 0, 1, 1]),
        ... )
        AccuracyResult(num_correct_predictions=5, num_predictions=5)
        >>> # with lists
        >>> accuracy(y_true=[1, 0, 0, 1, 1], y_pred=[1, 0, 1, 1, 1])
        AccuracyResult(num_correct_predictions=4, num_predictions=5)
        >>> # with polars Series
        >>> accuracy(
        ...     y_true=pl.Series("y_true", [1, 0, 0, 1, 1]),
        ...     y_pred=pl.Series("y_pred", [1, 0, 0, 1, 1]),
        ... )
        AccuracyResult(num_correct_predictions=5, num_predictions=5)
        >>> # with string labels
        >>> accuracy(
        ...     y_true=["cat", "dog", "cat", "dog"],
        ...     y_pred=["cat", "dog", "dog", "dog"],
        ... )
        AccuracyResult(num_correct_predictions=3, num_predictions=4)
        >>> # with missing values and missing_policy='propagate' (default)
        >>> accuracy(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ... )
        AccuracyResult(num_correct_predictions=nan, num_predictions=5)
        >>> # with missing values and missing_policy='omit'
        >>> accuracy(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ...     missing_policy="omit",
        ... )
        AccuracyResult(num_correct_predictions=4, num_predictions=4)

        ```
    """
    if isinstance(y_true, pl.Series) and isinstance(y_pred, pl.Series):
        return _accuracy_series(y_true=y_true, y_pred=y_pred, missing_policy=missing_policy)
    return _accuracy_array(y_true=y_true, y_pred=y_pred, missing_policy=missing_policy)


def _accuracy_core(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
    y_true_missing: bool,
    y_pred_missing: bool,
    num_predictions: int,
) -> AccuracyResult:
    r"""Compute the accuracy result from preprocessed inputs.

    Args:
        y_true: The preprocessed ground truth target labels.
        y_pred: The preprocessed predicted labels.
        y_true_missing: Whether ``y_true`` contains missing values.
        y_pred_missing: Whether ``y_pred`` contains missing values.
        num_predictions: The total number of predictions.

    Returns:
        The accuracy result.
    """
    num_correct_predictions = float("nan")
    if num_predictions > 0 and not y_true_missing and not y_pred_missing:
        num_correct_predictions = int((y_true == y_pred).sum())
    return AccuracyResult(
        num_correct_predictions=num_correct_predictions,
        num_predictions=num_predictions,
    )


def _accuracy_array(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    missing_policy: MissingPolicy = "propagate",
) -> AccuracyResult:
    r"""Compute the accuracy score for array-like inputs.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``.

    Returns:
        The accuracy result.

    Raises:
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` contains missing values and
            ``missing_policy`` is ``'raise'``.
        ValueError: if ``y_pred`` contains missing values and
            ``missing_policy`` is ``'raise'``.
    """
    y_true, y_pred = array.preprocess_pred(
        y_true=to_numpy_1d(y_true),
        y_pred=to_numpy_1d(y_pred),
        drop_missing=missing_policy == "omit",
    )
    # When missing_policy is 'propagate' or 'raise', check for missing
    # values in the (unfiltered) arrays. When 'omit', missing rows have
    # already been dropped so these checks always return False.
    return _accuracy_core(
        y_true=y_true,
        y_pred=y_pred,
        y_true_missing=array.contains_missing(
            arr=y_true, missing_policy=missing_policy, name="'y_true'"
        ),
        y_pred_missing=array.contains_missing(
            arr=y_pred, missing_policy=missing_policy, name="'y_pred'"
        ),
        num_predictions=y_true.size,
    )


def _accuracy_series(
    y_true: pl.Series,
    y_pred: pl.Series,
    *,
    missing_policy: MissingPolicy = "propagate",
) -> AccuracyResult:
    r"""Compute the accuracy score for Polars Series.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``.

    Returns:
        The accuracy result.

    Raises:
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` contains missing values and
            ``missing_policy`` is ``'raise'``.
        ValueError: if ``y_pred`` contains missing values and
            ``missing_policy`` is ``'raise'``.
    """
    y_true, y_pred = series.preprocess_pred(
        y_true=y_true,
        y_pred=y_pred,
        drop_missing=missing_policy == "omit",
    )
    # When missing_policy is 'propagate' or 'raise', check for missing
    # values in the (unfiltered) series. When 'omit', missing rows have
    # already been dropped so these checks always return False.
    return _accuracy_core(
        y_true=y_true,
        y_pred=y_pred,
        y_true_missing=series.contains_missing(series=y_true, missing_policy=missing_policy),
        y_pred_missing=series.contains_missing(series=y_pred, missing_policy=missing_policy),
        num_predictions=y_true.len(),
    )
