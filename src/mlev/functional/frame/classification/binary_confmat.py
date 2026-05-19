r"""Compute binary confusion matrix from array-like inputs.

This module powers :func:`mlev.functional.array.binary_confusion_matrix` and supports
Polars DataFrames.
"""

from __future__ import annotations

__all__ = ["binary_confusion_matrix"]

from typing import TYPE_CHECKING

from mlev.functional.array.classification.binary_confmat import compute_confusion_matrix
from mlev.utils.frame import contains_missing, preprocess
from mlev.utils.missing import MissingPolicy, check_missing_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from mlev.results import BinaryConfusionMatrixResult


def binary_confusion_matrix(
    frame: pl.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    *,
    betas: Sequence[float] = (1.0,),
    missing_policy: MissingPolicy = "propagate",
) -> BinaryConfusionMatrixResult:
    r"""Compute the binary confusion matrix.

    The positive class is the larger of the two unique label values
    (e.g. ``1`` vs ``0``, ``True`` vs ``False``, ``'dog'`` vs
    ``'cat'``).

    Args:
        frame: The input DataFrame containing the ground truth and
            predicted label columns.
        y_true_col: The name of the column containing the ground
            truth target labels.
        y_pred_col: The name of the column containing the predicted
            labels.
        betas: The beta values for F-beta score computation.
            Defaults to ``(1.0,)`` which gives the F1 score.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``.

    Returns:
        A ``BinaryConfusionMatrixResult`` with true/false
        positive/negative counts and derived metrics.

    Raises:
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` contains missing values and
            ``missing_policy`` is ``'raise'``.
        ValueError: if ``y_pred`` contains missing values and
            ``missing_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from mlev.functional.frame import binary_confusion_matrix
        >>> binary_confusion_matrix(
        ...     pl.DataFrame(
        ...         {
        ...             "y_true": [1, 0, 1, 1, 0, 0, 1, 0],
        ...             "y_pred": [1, 0, 1, 0, 1, 0, 1, 0],
        ...         }
        ...     ),
        ...     y_true_col="y_true",
        ...     y_pred_col="y_pred",
        ... )
        BinaryConfusionMatrixResult(true_positives=3, true_negatives=3, false_positives=1, false_negatives=1, ...)
        >>> binary_confusion_matrix(
        ...     pl.DataFrame(
        ...         {
        ...             "y_true": ["cat", "dog", "cat", "dog"],
        ...             "y_pred": ["cat", "dog", "dog", "dog"],
        ...         }
        ...     ),
        ...     y_true_col="y_true",
        ...     y_pred_col="y_pred",
        ... )
        BinaryConfusionMatrixResult(true_positives=2, true_negatives=1, false_positives=1, false_negatives=0, ...)
        >>> binary_confusion_matrix(
        ...     pl.DataFrame(
        ...         {
        ...             "y_true": [1.0, 0.0, 1.0, None],
        ...             "y_pred": [1.0, 0.0, 0.0, 1.0],
        ...         }
        ...     ),
        ...     y_true_col="y_true",
        ...     y_pred_col="y_pred",
        ...     missing_policy="omit",
        ... )
        BinaryConfusionMatrixResult(true_positives=1, true_negatives=1, false_positives=0, false_negatives=1, ...)

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
    return compute_confusion_matrix(
        y_true=frame[y_true_col],
        y_pred=frame[y_pred_col],
        has_missing=has_missing,
        betas=betas,
    )
