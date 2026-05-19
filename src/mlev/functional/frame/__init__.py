r"""DataFrame-based functional metrics.

This namespace exposes metric functions that operate on Polars DataFrames.

Example:
    ```pycon
    >>> import polars as pl
    >>> from mlev.functional.frame import accuracy
    >>> frame = pl.DataFrame({"target": [1, 0, 1], "pred": [1, 1, 1]})
    >>> accuracy(frame, y_true_col="target", y_pred_col="pred")
    AccuracyResult(num_correct_predictions=2, num_predictions=3)

    ```
"""

from __future__ import annotations

__all__ = ["accuracy"]

from mlev.functional.frame.classification.accuracy import accuracy
