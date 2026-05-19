r"""Reusable validation and integration helpers used across :mod:`mlev`.

Use this package for lower-level operations such as missing-value handling,
array conversion, logging configuration, and optional dependency checks.

Example:
    ```pycon
    >>> import numpy as np
    >>> from mlev.utils.array import contains_missing
    >>> bool(contains_missing(np.array([1.0, 2.0, float("nan")])))
    True

    ```
"""
