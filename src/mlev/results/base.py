r"""Define the base class for all results."""

from __future__ import annotations

__all__ = ["BaseResult"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.equality.tester import EqualNanEqualityTester, get_default_registry

if TYPE_CHECKING:
    from typing import Self


class BaseResult(ABC):
    r"""Base class for all results.

    Example:
        ```pycon
        >>> from mlev.results import AccuracyResult
        >>> m = AccuracyResult(num_correct_predictions=7, num_predictions=10)
        >>> m
        AccuracyResult(num_correct_predictions=7, num_predictions=10)
        >>> m.to_dict()
        {'accuracy': 0.7, 'num_correct_predictions': 7, 'num_predictions': 10}

        ```
    """

    @abstractmethod
    def combine(self, other: Self) -> Self:
        r"""Return the combined result of the two objects.

        Args:
            other: The value to combine with.

        Returns:
            The combined result of the two objects.

        Example:
            ```pycon
            >>> from mlev.results import AccuracyResult
            >>> m1 = AccuracyResult(num_correct_predictions=7, num_predictions=10)
            >>> m2 = AccuracyResult(num_correct_predictions=3, num_predictions=10)
            >>> m = m1.combine(m2)
            >>> m
            AccuracyResult(num_correct_predictions=10, num_predictions=20)
            >>> print(m.to_str())
            [██████████░░░░░░░░░░]  0.5000  (10/20)
            >>> m.to_dict()
            {'accuracy': 0.5, 'num_correct_predictions': 10, 'num_predictions': 20}

            ```
        """

    @abstractmethod
    def allclose(
        self,
        other: object,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        r"""Indicate whether two objects are equal within a tolerance.

        Args:
            other: The object to be compared with.
            rtol: The relative tolerance parameter. Must be non-negative.
            atol: The absolute tolerance parameter. Must be non-negative.
            equal_nan: If ``True``, then two ``NaN``s  will be considered
                as equal.

        Returns:
            ``True`` if the two objects are (element-wise) equal within a
                tolerance, otherwise ``False``

        Example:
            ```pycon
            >>> from mlev.results import AccuracyResult
            >>> m1 = AccuracyResult(num_correct_predictions=7, num_predictions=10)
            >>> m2 = AccuracyResult(num_correct_predictions=7, num_predictions=10)
            >>> m3 = AccuracyResult(num_correct_predictions=5, num_predictions=10)
            >>> m1.allclose(m2)
            True
            >>> m1.allclose(m3)
            False

            ```
        """

    @abstractmethod
    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``

        Example:
            ```pycon
            >>> from mlev.results import AccuracyResult
            >>> m1 = AccuracyResult(num_correct_predictions=7, num_predictions=10)
            >>> m2 = AccuracyResult(num_correct_predictions=7, num_predictions=10)
            >>> m3 = AccuracyResult(num_correct_predictions=5, num_predictions=10)
            >>> m1.equal(m2)
            True
            >>> m1.equal(m3)
            False

            ```
        """

    @abstractmethod
    def to_dict(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        r"""Return a dictionary representation of the result.

        Args:
            prefix: The prefix to add to each key of the result.
            suffix: The suffix to add to each key of the result.

        Returns:
            The dictionary representation of the result.

        Example:
        ```pycon
        >>> from mlev.results import AccuracyResult
        >>> m = AccuracyResult(num_correct_predictions=7, num_predictions=10)
        >>> m.to_dict()
        {'accuracy': 0.7, 'num_correct_predictions': 7, 'num_predictions': 10}

        ```
        """


get_default_registry().register(BaseResult, EqualNanEqualityTester(), exist_ok=True)
