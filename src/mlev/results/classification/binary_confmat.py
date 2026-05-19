r"""Classification accuracy result implementation."""

from __future__ import annotations

__all__ = [
    "BinaryConfusionMatrixResult",
    "check_betas",
    "compute_accuracy",
    "compute_f_beta_score",
    "compute_precision",
    "compute_recall",
    "compute_specificity",
]

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.format import make_bar

from mlev.results.base import BaseResult

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFUSION_MATRIX_ATTRS = (
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
)


def check_betas(betas: Sequence[float]) -> None:
    r"""Check the beta values are positive.

    Args:
        betas: The beta values to check.
    """
    for beta in betas:
        if beta < 0:
            msg = f"beta values must be >= 0, got {beta}"
            raise ValueError(msg)


def compute_accuracy(num_correct_predictions: int, num_predictions: int) -> float:
    r"""Compute the accuracy score.

    Args:
        num_correct_predictions: The number of correct predictions.
        num_predictions: The total number of predictions.

    Returns:
        The ratio ``num_correct_predictions / num_predictions``.
        Returns ``nan`` when ``num_predictions`` is ``0``.

    Example:
        ```pycon
        >>> from mlev.results.classification.binary_confmat import compute_accuracy
        >>> compute_accuracy(num_correct_predictions=7, num_predictions=10)
        0.7
        >>> compute_accuracy(num_correct_predictions=0, num_predictions=0)
        nan

        ```
    """
    if num_predictions == 0:
        return float("nan")
    return num_correct_predictions / num_predictions


def compute_precision(true_positives: int, false_positives: int, num_predictions: int) -> float:
    r"""Compute the precision score.

    Precision measures the proportion of true positives among all
    positive predictions.

    Args:
        true_positives: The number of true positives.
        false_positives: The number of false positives.
        num_predictions: The total number of predictions.

    Returns:
        The ratio ``true_positives / (true_positives + false_positives)``.
        Returns ``nan`` when ``num_predictions`` is ``0``, and ``0.0``
        when ``true_positives + false_positives`` is ``0``.

    Example:
        ```pycon
        >>> from mlev.results.classification.binary_confmat import compute_precision
        >>> compute_precision(true_positives=3, false_positives=1, num_predictions=10)
        0.75
        >>> compute_precision(true_positives=0, false_positives=0, num_predictions=10)
        0.0
        >>> compute_precision(true_positives=0, false_positives=0, num_predictions=0)
        nan

        ```
    """
    if num_predictions == 0:
        return float("nan")
    denominator = true_positives + false_positives
    return true_positives / denominator if denominator > 0 else 0.0


def compute_recall(true_positives: int, false_negatives: int, num_predictions: int) -> float:
    r"""Compute the recall (sensitivity) score.

    Recall measures the proportion of actual positives that are
    correctly identified.

    Args:
        true_positives: The number of true positives.
        false_negatives: The number of false negatives.
        num_predictions: The total number of predictions.

    Returns:
        The ratio ``true_positives / (true_positives + false_negatives)``.
        Returns ``nan`` when ``num_predictions`` is ``0``, and ``0.0``
        when ``true_positives + false_negatives`` is ``0``.

    Example:
        ```pycon
        >>> from mlev.results.classification.binary_confmat import compute_recall
        >>> compute_recall(true_positives=3, false_negatives=2, num_predictions=10)
        0.6
        >>> compute_recall(true_positives=0, false_negatives=0, num_predictions=10)
        0.0
        >>> compute_recall(true_positives=0, false_negatives=0, num_predictions=0)
        nan

        ```
    """
    if num_predictions == 0:
        return float("nan")
    denominator = true_positives + false_negatives
    return true_positives / denominator if denominator > 0 else 0.0


def compute_specificity(true_negatives: int, false_positives: int, num_predictions: int) -> float:
    r"""Compute the specificity (true negative rate) score.

    Specificity measures the proportion of actual negatives that are
    correctly identified.

    Args:
        true_negatives: The number of true negatives.
        false_positives: The number of false positives.
        num_predictions: The total number of predictions.

    Returns:
        The ratio ``true_negatives / (true_negatives + false_positives)``.
        Returns ``nan`` when ``num_predictions`` is ``0``, and ``0.0``
        when ``true_negatives + false_positives`` is ``0``.

    Example:
        ```pycon
        >>> from mlev.results.classification.binary_confmat import compute_specificity
        >>> compute_specificity(true_negatives=4, false_positives=1, num_predictions=10)
        0.8
        >>> compute_specificity(true_negatives=0, false_positives=0, num_predictions=10)
        0.0
        >>> compute_specificity(true_negatives=0, false_positives=0, num_predictions=0)
        nan

        ```
    """
    if num_predictions == 0:
        return float("nan")
    denominator = true_negatives + false_positives
    return true_negatives / denominator if denominator > 0 else 0.0


def compute_f_beta_score(precision: float, recall: float, beta: float) -> float:
    r"""Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and
    recall. ``beta=1`` gives equal weight to precision and recall
    (F1 score), ``beta<1`` weights precision more, and ``beta>1``
    weights recall more.

    Args:
        precision: The precision score.
        recall: The recall score.
        beta: The beta value. Must be non-negative.

    Returns:
        The F-beta score. Returns ``nan`` when either ``precision``
        or ``recall`` is ``nan``. Returns ``0.0`` when both
        ``precision`` and ``recall`` are ``0.0``.

    Raises:
        ValueError: if ``beta`` is negative.

    Example:
        ```pycon
        >>> from mlev.results.classification.binary_confmat import compute_f_beta_score
        >>> compute_f_beta_score(precision=0.75, recall=0.6, beta=1.0)
        0.6666666666666665
        >>> compute_f_beta_score(precision=0.75, recall=0.6, beta=0.5)
        0.7142857142857143
        >>> compute_f_beta_score(precision=0.75, recall=0.6, beta=2.0)
        0.625
        >>> compute_f_beta_score(precision=0.0, recall=0.0, beta=1.0)
        0.0

        ```
    """
    if beta < 0:
        msg = f"beta must be >= 0, got {beta}"
        raise ValueError(msg)
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    beta_sq = beta**2
    denominator = beta_sq * precision + recall
    return (1 + beta_sq) * (precision * recall) / denominator if denominator > 0 else 0.0


def f_beta_label(beta: float) -> str:
    return f"F{int(beta)}" if beta == int(beta) else f"F{beta:g}"


@dataclass(frozen=True)
class BinaryConfusionMatrixResult(BaseResult):
    r"""Store aggregated values from a binary confusion matrix used to
    compute classification metrics including accuracy, precision,
    recall, specificity, and F-beta scores.

    Use :meth:`from_confusion_matrix` to construct an instance from
    raw confusion matrix counts.

    Attributes:
        true_positives: The number of true positives.
        true_negatives: The number of true negatives.
        false_positives: The number of false positives.
        false_negatives: The number of false negatives.
        num_predictions: The total number of predictions.
        num_correct_predictions: The number of correct predictions.
        accuracy: The accuracy score.
        precision: The precision score.
        recall: The recall score.
        specificity: The specificity score.
        f_beta_scores: A mapping of beta values to F-beta scores.

    Example:
        ```pycon
        >>> from mlev.results import BinaryConfusionMatrixResult
        >>> m = BinaryConfusionMatrixResult.from_confusion_matrix(
        ...     true_positives=3,
        ...     true_negatives=4,
        ...     false_positives=1,
        ...     false_negatives=2,
        ... )
        >>> m.accuracy
        0.7
        >>> m.precision
        0.75
        >>> m.recall
        0.6
        >>> m.specificity
        0.8
        >>> m.f_beta_scores
        {1.0: 0.6666666666666665}
        >>> m.to_dict()
        {'accuracy': 0.7, 'precision': 0.75, 'recall': 0.6, 'specificity': 0.8, 'f1': 0.6666666666666665, 'num_correct_predictions': 7, 'num_predictions': 10, 'true_positives': 3, 'true_negatives': 4, 'false_positives': 1, 'false_negatives': 2}

        ```
    """

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    num_predictions: int
    num_correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f_beta_scores: dict[float, float]

    def combine(self, other: BinaryConfusionMatrixResult) -> BinaryConfusionMatrixResult:
        if not isinstance(other, BinaryConfusionMatrixResult):
            msg = f"Cannot combine {self.__class__.__qualname__} with {type(other)}"
            raise TypeError(msg)
        return BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=self.true_positives + other.true_positives,
            true_negatives=self.true_negatives + other.true_negatives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
            betas=list(self.f_beta_scores.keys()),
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
            asdict(self), asdict(other), atol=atol, rtol=rtol, equal_nan=equal_nan
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(asdict(self), asdict(other), equal_nan=equal_nan)

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        out: dict[str, int | float] = {
            f"{prefix}accuracy{suffix}": self.accuracy,
            f"{prefix}precision{suffix}": self.precision,
            f"{prefix}recall{suffix}": self.recall,
            f"{prefix}specificity{suffix}": self.specificity,
        }
        for beta, score in self.f_beta_scores.items():
            key = f"f{int(beta)}" if beta == int(beta) else f"f{beta:g}"
            out[f"{prefix}{key}{suffix}"] = score
        out.update(
            {
                f"{prefix}num_correct_predictions{suffix}": self.num_correct_predictions,
                f"{prefix}num_predictions{suffix}": self.num_predictions,
                f"{prefix}true_positives{suffix}": self.true_positives,
                f"{prefix}true_negatives{suffix}": self.true_negatives,
                f"{prefix}false_positives{suffix}": self.false_positives,
                f"{prefix}false_negatives{suffix}": self.false_negatives,
            }
        )
        return out

    def to_str(self) -> str:
        r"""Return a human-friendly text representation of the
        classification results.

        Returns:
            A formatted string with a confusion matrix summary and
            progress bars for each metric.

        Example:
            ```pycon
            >>> from mlev.results import BinaryConfusionMatrixResult
            >>> m = BinaryConfusionMatrixResult.from_confusion_matrix(
            ...     true_positives=3,
            ...     true_negatives=4,
            ...     false_positives=1,
            ...     false_negatives=2,
            ... )
            >>> print(m.to_str())
            Binary Confusion Matrix
            -----------------------
            n=10  TP=3  TN=4  FP=1  FN=2
            Accuracy    [██████████████░░░░░░]  0.7000  (7/10)
            Precision   [███████████████░░░░░]  0.7500  (3/4)
            Recall      [████████████░░░░░░░░]  0.6000  (3/5)
            Specificity [████████████████░░░░]  0.8000  (4/5)
            F1          [█████████████░░░░░░░]  0.6667

            ```
        """
        header = "Binary Confusion Matrix"
        separator = "-" * len(header)
        summary = (
            f"n={self.num_predictions:,}  "
            f"TP={self.true_positives:,}  "
            f"TN={self.true_negatives:,}  "
            f"FP={self.false_positives:,}  "
            f"FN={self.false_negatives:,}"
        )

        # Each entry: (label, value, optional (numerator, denominator) for count suffix)
        tp, tn, fp, fn = (
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        )
        metrics: list[tuple[str, float, tuple[int, int] | None]] = [
            ("Accuracy", self.accuracy, (self.num_correct_predictions, self.num_predictions)),
            ("Precision", self.precision, (tp, tp + fp)),
            ("Recall", self.recall, (tp, tp + fn)),
            ("Specificity", self.specificity, (tn, tn + fp)),
            *[(f_beta_label(beta), score, None) for beta, score in self.f_beta_scores.items()],
        ]

        metric_lines = []
        for name, value, counts in metrics:
            line = f"{name:<11} {make_bar(value, length=20)}  {value:.4f}"
            if counts is not None:
                numerator, denominator = counts
                line += f"  ({numerator:,}/{denominator:,})"
            metric_lines.append(line)

        metric_text = "\n".join(metric_lines)
        return f"{header}\n{separator}\n{summary}\n{metric_text}"

    @classmethod
    def from_confusion_matrix(
        cls,
        true_positives: int,
        true_negatives: int,
        false_positives: int,
        false_negatives: int,
        betas: Sequence[float] = (1.0,),
    ) -> BinaryConfusionMatrixResult:
        r"""Create a result from raw confusion matrix counts.

        Args:
            true_positives: The number of true positives.
            true_negatives: The number of true negatives.
            false_positives: The number of false positives.
            false_negatives: The number of false negatives.
            betas: The beta values for F-beta score computation.
                Defaults to ``(1.0,)`` which gives the F1 score.

        Returns:
            A fully populated ``BinaryConfusionMatrixResult``.

        Raises:
            ValueError: if any count is negative.
            ValueError: if any beta value is negative.

        Example:
            ```pycon
            >>> from mlev.results import BinaryConfusionMatrixResult
            >>> m = BinaryConfusionMatrixResult.from_confusion_matrix(
            ...     true_positives=3,
            ...     true_negatives=4,
            ...     false_positives=1,
            ...     false_negatives=2,
            ...     betas=[0.5, 1.0, 2.0],
            ... )
            >>> m.f_beta_scores
            {0.5: 0.7142857142857143, 1.0: 0.6666666666666665, 2.0: 0.625}

            ```
        """
        for name, value in (
            ("true_positives", true_positives),
            ("true_negatives", true_negatives),
            ("false_positives", false_positives),
            ("false_negatives", false_negatives),
        ):
            if value < 0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)
        check_betas(betas)

        num_predictions = true_positives + true_negatives + false_positives + false_negatives
        num_correct_predictions = true_positives + true_negatives
        precision = compute_precision(true_positives, false_positives, num_predictions)
        recall = compute_recall(true_positives, false_negatives, num_predictions)

        return cls(
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            num_predictions=num_predictions,
            num_correct_predictions=num_correct_predictions,
            accuracy=compute_accuracy(num_correct_predictions, num_predictions),
            precision=precision,
            recall=recall,
            specificity=compute_specificity(true_negatives, false_positives, num_predictions),
            f_beta_scores={beta: compute_f_beta_score(precision, recall, beta) for beta in betas},
        )
