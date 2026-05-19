r"""Compute classification accuracy from DataFrame columns.

This module powers :func:`mlev.functional.frame.accuracy` and evaluates a pair
of ground-truth and predicted-label columns from a Polars DataFrame.
"""

from __future__ import annotations

__all__ = ["accuracy"]

from typing import TYPE_CHECKING

from mlev.results import AccuracyResult
from mlev.utils.frame import contains_missing, preprocess
from mlev.utils.missing import MissingPolicy, check_missing_policy

if TYPE_CHECKING:
    import polars as pl


def accuracy(
    frame: pl.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    *,
    missing_policy: MissingPolicy = "propagate",
) -> AccuracyResult:
    r"""Compute the accuracy score between two columns of a DataFrame.

    Args:
        frame: The input DataFrame containing the ground truth and
            predicted label columns.
        y_true_col: The name of the column containing the ground
            truth target labels.
        y_pred_col: The name of the column containing the predicted
            labels.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``.

    Returns:
        The accuracy result. When missing values are present and
            ``missing_policy="propagate"``, the result keeps
            ``num_predictions`` and sets ``num_correct_predictions`` to
            ``nan``.

    Raises:
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if the DataFrame contains missing values in
            ``y_true_col`` or ``y_pred_col`` and ``missing_policy``
            is ``'raise'``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.functional.frame import accuracy
        >>> frame = pl.DataFrame({"y_true": [1, 0, 0, 1, 1], "y_pred": [1, 0, 0, 1, 0]})
        >>> accuracy(frame, y_true_col="y_true", y_pred_col="y_pred")
        AccuracyResult(num_correct_predictions=4, num_predictions=5)
        >>> frame = pl.DataFrame({"y_true": [1, 0, 0, 1, None], "y_pred": [1, 0, 0, 1, 0]})
        >>> accuracy(frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="omit")
        AccuracyResult(num_correct_predictions=4, num_predictions=4)

        ```
    """
    check_missing_policy(missing_policy)
    frame = preprocess(
        frame.select([y_true_col, y_pred_col]),
        drop_missing=missing_policy == "omit",
    )
    # When missing_policy is 'propagate' or 'raise', check for missing
    # values in the (unfiltered) frame. When 'omit', missing rows have
    # already been dropped so this check always returns False.
    has_missing = contains_missing(frame, missing_policy=missing_policy)

    num_predictions = len(frame)
    num_correct_predictions = float("nan")
    if num_predictions > 0 and not has_missing:
        num_correct_predictions = int((frame[y_true_col] == frame[y_pred_col]).sum())
    return AccuracyResult(
        num_correct_predictions=num_correct_predictions,
        num_predictions=num_predictions,
    )
