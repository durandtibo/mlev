r"""Classification accuracy result implementation."""

from __future__ import annotations

__all__ = ["BinaryConfusionMatrixResult"]

from dataclasses import dataclass

from coola.equality import objects_are_allclose, objects_are_equal

from mlev.results.base import BaseResult


@dataclass(frozen=True)
class BinaryConfusionMatrixResult(BaseResult):
    r"""Store aggregated values from a binary confusion matrix used to
    compute classification accuracy.

        The number of predictions is the sum of true positives, true
        negatives, false positives, and false negatives.

    Attributes:
        true_positives: The number of true positives (correctly
            predicted positive class).
        true_negatives: The number of true negatives (correctly
            predicted negative class).
        false_positives: The number of false positives (negative
            class incorrectly predicted as positive).
        false_negatives: The number of false negatives (positive
            class incorrectly predicted as negative).

    Example:
    ```pycon
    >>> from mlev.results import BinaryConfusionMatrixResult
    >>> m = BinaryConfusionMatrixResult(
    ...     true_positives=3,
    ...     true_negatives=4,
    ...     false_positives=1,
    ...     false_negatives=2,
    ... )
    >>> m
    BinaryConfusionMatrixResult(true_positives=3, true_negatives=4, false_positives=1, false_negatives=2)
    >>> m.to_dict()
    {'accuracy': 0.7, 'num_correct_predictions': 7, 'num_predictions': 10, 'true_positives': 3, 'true_negatives': 4, 'false_positives': 1, 'false_negatives': 2}

    ```
    """

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def __post_init__(self) -> None:
        r"""Validate confusion matrix counts after dataclass
        initialization.

        Raises:
            ValueError: if any of the four counts is negative.
        """
        for attr in ("true_positives", "true_negatives", "false_positives", "false_negatives"):
            value = getattr(self, attr)
            if value < 0:
                msg = f"{attr} must be >= 0, got {value}"
                raise ValueError(msg)

    @property
    def num_correct_predictions(self) -> int:
        r"""Return the number of correct predictions.

        Returns:
            The sum of true positives and true negatives.
        """
        return self.true_positives + self.true_negatives

    @property
    def num_predictions(self) -> int:
        r"""Return the total number of predictions.

        Returns:
            The sum of all four confusion matrix counts.
        """
        return (
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        )

    @property
    def accuracy(self) -> float:
        r"""Return the accuracy ratio.

        Returns:
            The ratio ``num_correct_predictions / num_predictions``.
            Returns ``nan`` when ``num_predictions`` is ``0``.
        """
        if self.num_predictions == 0:
            return float("nan")
        return self.num_correct_predictions / self.num_predictions

    def combine(self, other: BinaryConfusionMatrixResult) -> BinaryConfusionMatrixResult:
        r"""Combine two results by summing their confusion matrix counts.

        Args:
            other: The other result to combine with.

        Returns:
            A new result with summed counts.

        Raises:
            TypeError: if ``other`` is not a
                ``BinaryConfusionMatrixResult``.
        """
        if not isinstance(other, BinaryConfusionMatrixResult):
            msg = f"Cannot combine {self.__class__.__qualname__} with {type(other)}"
            raise TypeError(msg)
        return BinaryConfusionMatrixResult(
            true_positives=self.true_positives + other.true_positives,
            true_negatives=self.true_negatives + other.true_negatives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )

    def allclose(
        self,
        other: object,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        r"""Indicate if two results are approximately equal.

        Args:
            other: The object to compare with.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            equal_nan: If ``True``, ``NaN`` values are considered equal.

        Returns:
            ``True`` if all four counts are approximately equal.
        """
        if type(other) is not type(self):
            return False
        return all(
            objects_are_allclose(
                getattr(self, attr),
                getattr(other, attr),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
            for attr in ("true_positives", "true_negatives", "false_positives", "false_negatives")
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Indicate if two results are exactly equal.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, ``NaN`` values are considered equal.

        Returns:
            ``True`` if all four counts are equal.
        """
        if type(other) is not type(self):
            return False
        return all(
            objects_are_equal(getattr(self, attr), getattr(other, attr), equal_nan=equal_nan)
            for attr in ("true_positives", "true_negatives", "false_positives", "false_negatives")
        )

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        r"""Convert the result to a dictionary.

        Args:
            prefix: An optional prefix for all keys.
            suffix: An optional suffix for all keys.

        Returns:
            A dictionary with accuracy, correct predictions, total
            predictions, and all four confusion matrix counts.
        """
        return {
            f"{prefix}accuracy{suffix}": self.accuracy,
            f"{prefix}num_correct_predictions{suffix}": self.num_correct_predictions,
            f"{prefix}num_predictions{suffix}": self.num_predictions,
            f"{prefix}true_positives{suffix}": self.true_positives,
            f"{prefix}true_negatives{suffix}": self.true_negatives,
            f"{prefix}false_positives{suffix}": self.false_positives,
            f"{prefix}false_negatives{suffix}": self.false_negatives,
        }
