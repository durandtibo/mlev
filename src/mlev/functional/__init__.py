r"""Compute evaluation results with stateless functional helpers.

Use this package when you already have predictions and labels and want to
compute metrics without creating metric objects.

Example:
    ```pycon
    >>> import numpy as np
    >>> from mlev.functional.array import accuracy
    >>> accuracy(y_true=np.array([1, 0, 1]), y_pred=np.array([1, 1, 1]))
    AccuracyResult(num_correct_predictions=2, num_predictions=3)

    ```
"""
