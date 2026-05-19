r"""Balanced accuracy result implementation."""

from __future__ import annotations

__all__ = ["BalancedAccuracyResult", "compute_balanced_accuracy"]

import math
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

import numpy as np
from coola.equality import objects_are_allclose, objects_are_equal

from mlev.results.base import BaseResult
from mlev.utils.array import check_same_shape
from mlev.utils.format import make_robust_bar

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_balanced_accuracy(
    per_class_correct: np.ndarray,
    per_class_total: np.ndarray,
) -> float:
    r"""Compute the balanced accuracy score.

    Balanced accuracy is the macro-average of per-class recall:
    ``mean(correct_k / total_k for each class k)``, consistent with
    :func:`sklearn.metrics.balanced_accuracy_score`.

    Args:
        per_class_correct: The number of correctly predicted samples
            per class.
        per_class_total: The total number of actual samples per class.

    Returns:
        The balanced accuracy score in ``[0, 1]``. Returns ``nan``
        when any value is ``nan``, or when all class totals are ``0``
        (no predictions).

    Raises:
        ValueError: if ``per_class_correct`` and ``per_class_total``
            have different shapes.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.results.classification.balanced_accuracy import compute_balanced_accuracy
        >>> # binary: recall_pos=3/5=0.6, recall_neg=4/5=0.8
        >>> compute_balanced_accuracy(
        ...     per_class_correct=np.array([3, 4]),
        ...     per_class_total=np.array([5, 5]),
        ... )
        0.7
        >>> # multiclass: 3 classes
        >>> compute_balanced_accuracy(
        ...     per_class_correct=np.array([4, 3, 5]),
        ...     per_class_total=np.array([5, 4, 5]),
        ... )
        0.8833333333333333
        >>> # empty
        >>> compute_balanced_accuracy(
        ...     per_class_correct=np.array([0, 0]),
        ...     per_class_total=np.array([0, 0]),
        ... )
        nan

        ```
    """
    check_same_shape([per_class_correct, per_class_total])

    if np.any(np.isnan(per_class_correct)) or np.any(np.isnan(per_class_total)):
        return float("nan")

    mask = per_class_total > 0
    if not np.any(mask):
        return float("nan")

    recalls = per_class_correct[mask] / per_class_total[mask]
    return float(recalls.mean())


@dataclass(frozen=True)
class BalancedAccuracyResult(BaseResult):
    r"""Store aggregated values used to compute balanced classification
    accuracy.

    Balanced accuracy is the macro-average of per-class recall,
    consistent with :func:`sklearn.metrics.balanced_accuracy_score`.
    It works for binary and multi-class classification.

    The per-class correct and total counts are stored rather than the
    derived score so that :meth:`combine` can correctly recompute
    balanced accuracy from summed counts across batches.

    Attributes:
        per_class_correct: A 1D numpy array with the number of
            correctly predicted samples per class.
        per_class_total: A 1D numpy array with the total number of
            actual samples per class.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from mlev.results import BalancedAccuracyResult
        >>> # binary
        >>> m = BalancedAccuracyResult(
        ...     per_class_correct=np.array([3, 4]),
        ...     per_class_total=np.array([5, 5]),
        ... )
        >>> m.balanced_accuracy
        0.7
        >>> m.to_dict()
        {'balanced_accuracy': 0.7, 'per_class_correct': array([3, 4]), 'per_class_total': array([5, 5])}
        >>> # multiclass
        >>> m3 = BalancedAccuracyResult(
        ...     per_class_correct=np.array([4, 3, 5]),
        ...     per_class_total=np.array([5, 4, 5]),
        ... )
        >>> m3.balanced_accuracy
        0.8833333333333333

        ```
    """

    per_class_correct: np.ndarray
    per_class_total: np.ndarray

    def __post_init__(self) -> None:
        check_same_shape([self.per_class_correct, self.per_class_total])

        for name, arr in (
            ("per_class_correct", self.per_class_correct),
            ("per_class_total", self.per_class_total),
        ):
            invalid = arr[~np.isnan(arr.astype(float))] if arr.dtype.kind == "f" else arr
            if np.any(invalid < 0):
                neg_idx = int(np.argmax(invalid < 0))
                msg = f"{name}[{neg_idx}] must be >= 0, got {invalid[neg_idx]}"
                raise ValueError(msg)

    @property
    def num_classes(self) -> int:
        r"""Return the number of classes.

        Returns:
            The number of classes.
        """
        return len(self.per_class_correct)

    @property
    def num_predictions(self) -> int | float:
        r"""Return the total number of predictions.

        Returns:
            The sum of all per-class totals, or ``nan`` if any is
            ``nan``.
        """
        return float(self.per_class_total.sum())

    @property
    def balanced_accuracy(self) -> float:
        r"""Return the balanced accuracy score.

        Returns:
            The macro-average of per-class recall. Returns ``nan``
            when any count is ``nan`` or when all class totals are
            ``0``.
        """
        return compute_balanced_accuracy(
            per_class_correct=self.per_class_correct,
            per_class_total=self.per_class_total,
        )

    def combine(self, other: BalancedAccuracyResult) -> BalancedAccuracyResult:
        if not isinstance(other, BalancedAccuracyResult):
            msg = f"Cannot combine {self.__class__.__qualname__} with {type(other)}"
            raise TypeError(msg)
        if self.num_classes != other.num_classes:
            msg = (
                f"Cannot combine results with different number of classes: "
                f"{self.num_classes} vs {other.num_classes}"
            )
            raise ValueError(msg)
        return BalancedAccuracyResult(
            per_class_correct=self.per_class_correct + other.per_class_correct,
            per_class_total=self.per_class_total + other.per_class_total,
        )

    def allclose(
        self,
        other: object,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_allclose(
            asdict(self),
            asdict(other),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(asdict(self), asdict(other), equal_nan=equal_nan)

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float | np.ndarray]:
        return {
            f"{prefix}balanced_accuracy{suffix}": self.balanced_accuracy,
            f"{prefix}per_class_correct{suffix}": self.per_class_correct,
            f"{prefix}per_class_total{suffix}": self.per_class_total,
        }

    def to_display(self) -> str:
        score = self.balanced_accuracy
        bar = make_robust_bar(score, length=20)
        score_str = "nan" if math.isnan(score) else f"{score:.4f}"
        per_class = "  ".join(
            f"class{i}={int(c)}/{int(t)}"
            for i, (c, t) in enumerate(zip(self.per_class_correct, self.per_class_total))
        )
        return f"Balanced Accuracy {bar}  {score_str}  ({per_class})"

    @classmethod
    def from_predictions(
        cls,
        y_true: Sequence,
        y_pred: Sequence,
    ) -> BalancedAccuracyResult:
        r"""Create a result from ground truth and predicted labels.

        Args:
            y_true: The ground truth labels.
            y_pred: The predicted labels.

        Returns:
            A ``BalancedAccuracyResult`` with per-class numpy arrays.

        Example:
            ```pycon
            >>> from mlev.results import BalancedAccuracyResult
            >>> m = BalancedAccuracyResult.from_predictions(
            ...     y_true=[0, 1, 0, 1, 2, 2],
            ...     y_pred=[0, 1, 1, 1, 2, 0],
            ... )
            >>> m.balanced_accuracy
            0.7777777777777778

            ```
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        classes = np.unique(y_true_arr)
        per_class_correct = np.array(
            [int(np.sum((y_true_arr == cls) & (y_pred_arr == cls))) for cls in classes]
        )
        per_class_total = np.array([int(np.sum(y_true_arr == cls)) for cls in classes])
        return cls(
            per_class_correct=per_class_correct,
            per_class_total=per_class_total,
        )
