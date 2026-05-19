r"""Array-based functional metrics.

This namespace exposes metric functions that accept NumPy arrays, Polars
series, and other supported array-like inputs.

Example:
    ```pycon
    >>> from mlev.functional.array import accuracy
    >>> accuracy(y_true=[1, 0, 1], y_pred=[1, 1, 1])
    AccuracyResult(num_correct_predictions=2, num_predictions=3)

    ```
"""

from __future__ import annotations

__all__ = ["accuracy"]

from mlev.functional.array.classification.accuracy import accuracy
