r"""Evaluate machine-learning predictions with lightweight metric helpers.

The public API currently focuses on:

- stateless functional metric helpers in :mod:`mlev.functional`,
- immutable metric result containers in :mod:`mlev.results`.

Example:
    ```pycon
    >>> from mlev.functional.array import accuracy
    >>> result = accuracy(y_true=[1, 0, 1], y_pred=[1, 1, 1])
    >>> result
    AccuracyResult(num_correct_predictions=2, num_predictions=3)
    >>> result.accuracy
    0.6666666666666666

    ```
"""

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
