r"""Contain code to compute the accuracy metric."""

from __future__ import annotations

__all__ = ["accuracy"]

from typing import TYPE_CHECKING

import polars as pl

from mlev.results import AccuracyResult
from mlev.utils import array, series
from mlev.utils.array import to_numpy_1d

if TYPE_CHECKING:
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
    """
    if isinstance(y_true, pl.Series) and isinstance(y_pred, pl.Series):
        return _accuracy_series(y_true=y_true, y_pred=y_pred, missing_policy=missing_policy)
    return _accuracy_array(y_true=y_true, y_pred=y_pred, missing_policy=missing_policy)


def _accuracy_array(
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
    """
    y_true, y_pred = array.preprocess_pred(
        y_true=to_numpy_1d(y_true),
        y_pred=to_numpy_1d(y_pred),
        drop_missing=missing_policy == "omit",
    )

    # When missing_policy is 'propagate' or 'raise', check for missing
    # values in the (unfiltered) arrays. When 'omit', missing rows have
    # already been dropped so these checks always return False.
    y_true_missing = array.contains_missing(
        arr=y_true, missing_policy=missing_policy, name="'y_true'"
    )
    y_pred_missing = array.contains_missing(
        arr=y_pred, missing_policy=missing_policy, name="'y_pred'"
    )

    num_predictions = y_true.size
    num_correct_predictions = float("nan")
    if num_predictions > 0 and not y_true_missing and not y_pred_missing:
        num_correct_predictions = int((y_true == y_pred).sum())
    return AccuracyResult(
        num_correct_predictions=num_correct_predictions,
        num_predictions=num_predictions,
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
    y_true_missing = series.contains_missing(series=y_true, missing_policy=missing_policy)
    y_pred_missing = series.contains_missing(series=y_pred, missing_policy=missing_policy)

    num_predictions = y_true.len()
    num_correct_predictions = float("nan")
    if num_predictions > 0 and not y_true_missing and not y_pred_missing:
        num_correct_predictions = int((y_true == y_pred).sum())
    return AccuracyResult(
        num_correct_predictions=num_correct_predictions,
        num_predictions=num_predictions,
    )
