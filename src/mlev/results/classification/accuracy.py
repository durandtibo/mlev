r"""Contain the accuracy result."""

from __future__ import annotations

__all__ = ["AccuracyResult"]

from dataclasses import dataclass

from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.format import make_bar

from mlev.results.base import BaseResult


@dataclass(frozen=True)
class AccuracyResult(BaseResult):
    r"""Define the accuracy result.

    Attributes:
        num_correct_predictions: The number of correct predictions.
        num_predictions: The number of predictions.

    Example:
        ```pycon
        >>> from mlev.results.classification import AccuracyResult
        >>> m = AccuracyResult(num_correct_predictions=7, num_predictions=10)
        >>> m
        AccuracyResult(num_correct_predictions=7, num_predictions=10)
        >>> print(m.to_str())
        [██████████████░░░░░░]  0.7000  (7/10)
        >>> m.to_dict()
        {'accuracy': 0.7, 'num_correct_predictions': 7, 'num_predictions': 10}

        ```
    """

    num_correct_predictions: int
    num_predictions: int

    def __post_init__(self) -> None:
        if self.num_predictions < 0:
            msg = f"num_predictions must be >= 0, got {self.num_predictions}"
            raise ValueError(msg)
        if self.num_correct_predictions < 0:
            msg = f"num_correct_predictions must be >= 0, got {self.num_correct_predictions}"
            raise ValueError(msg)
        if self.num_correct_predictions > self.num_predictions:
            msg = (
                f"num_correct_predictions ({self.num_correct_predictions}) "
                f"cannot exceed num_predictions ({self.num_predictions})"
            )
            raise ValueError(msg)

    @property
    def accuracy(self) -> float:
        if self.num_predictions == 0:
            return float("nan")
        return self.num_correct_predictions / self.num_predictions

    def combine(self, other: AccuracyResult) -> AccuracyResult:
        if not isinstance(other, AccuracyResult):
            msg = f"Cannot combine {self.__class__.__qualname__} with {type(other)}"
            raise TypeError(msg)
        return AccuracyResult(
            num_correct_predictions=self.num_correct_predictions + other.num_correct_predictions,
            num_predictions=self.num_predictions + other.num_predictions,
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
            self.num_correct_predictions,
            other.num_correct_predictions,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ) and objects_are_allclose(
            self.num_predictions, other.num_predictions, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(
            self.num_correct_predictions, other.num_correct_predictions, equal_nan=equal_nan
        ) and objects_are_equal(self.num_predictions, other.num_predictions, equal_nan=equal_nan)

    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        return {
            f"{prefix}accuracy{suffix}": self.accuracy,
            f"{prefix}num_correct_predictions{suffix}": self.num_correct_predictions,
            f"{prefix}num_predictions{suffix}": self.num_predictions,
        }

    def to_str(self) -> str:
        if self.num_predictions == 0:
            return f"{self.__class__.__qualname__}: no predictions"
        accuracy = self.accuracy
        return f"{make_bar(accuracy, length=20)}  {accuracy:.4f}  ({self.num_correct_predictions:,}/{self.num_predictions:,})"
