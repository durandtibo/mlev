r"""Compute binary confusion matrix from array-like inputs.

This module powers :func:`mlev.functional.array.binary_confusion_matrix` and supports
NumPy arrays, Polars series, and Python sequences.
"""

from __future__ import annotations

__all__ = [
    "binary_confusion_matrix",
    "compute_confusion_matrix",
]

from typing import TYPE_CHECKING

import polars as pl
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from mlev.results import BinaryConfusionMatrixResult
from mlev.utils import array, series
from mlev.utils.array import to_numpy_1d
from mlev.utils.missing import MissingPolicy, check_missing_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from mlev.typing import ArrayLike


def binary_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    betas: Sequence[float] = (1.0,),
    missing_policy: MissingPolicy = "propagate",
) -> BinaryConfusionMatrixResult:
    r"""Compute the binary confusion matrix.

    The positive class is the larger of the two unique label values
    (e.g. ``1`` vs ``0``, ``True`` vs ``False``, ``'dog'`` vs
    ``'cat'``).

    Args:
        y_true: The ground truth target labels. Must contain at most
            two unique non-missing values.
        y_pred: The predicted labels. Must contain at most two unique
            non-missing values.
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
        >>> import numpy as np
        >>> import polars as pl
        >>> from mlev.functional.array import binary_confusion_matrix
        >>> binary_confusion_matrix(
        ...     y_true=np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        ...     y_pred=np.array([1, 0, 1, 0, 1, 0, 1, 0]),
        ... )
        BinaryConfusionMatrixResult(true_positives=3, true_negatives=3, false_positives=1, false_negatives=1, ...)
        >>> binary_confusion_matrix(
        ...     y_true=["cat", "dog", "cat", "dog"],
        ...     y_pred=["cat", "dog", "dog", "dog"],
        ... )
        BinaryConfusionMatrixResult(true_positives=2, true_negatives=1, false_positives=1, false_negatives=0, ...)
        >>> binary_confusion_matrix(
        ...     y_true=np.array([1.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        ...     missing_policy="omit",
        ... )
        BinaryConfusionMatrixResult(true_positives=1, true_negatives=1, false_positives=0, false_negatives=1, ...)

        ```
    """
    check_missing_policy(missing_policy)

    if isinstance(y_true, pl.Series) and isinstance(y_pred, pl.Series):
        y_true, y_pred = series.preprocess([y_true, y_pred], drop_missing=missing_policy == "omit")
        has_missing = series.contains_missing(
            y_true, missing_policy=missing_policy
        ) or series.contains_missing(y_pred, missing_policy=missing_policy)
        return compute_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            has_missing=has_missing,
            betas=betas,
        )

    y_true, y_pred = array.preprocess_1d(
        [to_numpy_1d(y_true), to_numpy_1d(y_pred)],
        drop_missing=missing_policy == "omit",
    )
    has_missing = array.contains_missing(
        y_true, missing_policy=missing_policy, name="'y_true'"
    ) or array.contains_missing(y_pred, missing_policy=missing_policy, name="'y_pred'")
    return compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        has_missing=has_missing,
        betas=betas,
    )


def compute_confusion_matrix(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
    *,
    has_missing: bool,
    betas: Sequence[float],
) -> BinaryConfusionMatrixResult:
    r"""Compute a ``BinaryConfusionMatrixResult`` from pre-processed
    label arrays.

    The positive class is the larger of the two unique label values.
    If only one class is present, it is assumed to be the positive
    class. Handles empty arrays, single-class arrays, and missing
    value propagation.

    Args:
        y_true: The ground truth labels, already preprocessed
            (missing rows dropped if ``omit`` policy was used).
        y_pred: The predicted labels, already preprocessed.
        has_missing: If ``True``, all confusion matrix counts are set
            to ``nan`` to propagate missing values.
        betas: The beta values for F-beta score computation.

    Returns:
        A ``BinaryConfusionMatrixResult`` with true/false
        positive/negative counts and derived metrics.
    """
    if has_missing:
        return BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=float("nan"),
            true_negatives=float("nan"),
            false_positives=float("nan"),
            false_negatives=float("nan"),
            betas=betas,
        )

    if len(y_true) == 0:
        return BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            betas=betas,
        )

    cm = sklearn_confusion_matrix(y_true=y_true, y_pred=y_pred)

    if cm.shape == (1, 1):
        # Only one class present — assume it is the positive class.
        # tp = correctly predicted positives, fn = missed positives.
        # Since there are no negatives, tn=0 and fp=0.
        count = int(cm[0, 0])
        return BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=count,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            betas=betas,
        )

    tn, fp, fn, tp = cm.ravel()
    return BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        betas=betas,
    )
