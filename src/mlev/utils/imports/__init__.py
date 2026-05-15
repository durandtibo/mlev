r"""Helpers for optional dependencies used by mlev.

The utilities in this package let callers:

- check whether an optional package is installed,
- fail early with a clear error when a package is required,
- gate function execution behind package availability.
"""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "check_rich",
    "check_sklearn",
    "colorlog_available",
    "is_colorlog_available",
    "is_rich_available",
    "is_sklearn_available",
    "raise_colorlog_missing_error",
    "raise_rich_missing_error",
    "raise_sklearn_missing_error",
    "rich_available",
    "sklearn_available",
]

from mlev.utils.imports.colorlog import (
    check_colorlog,
    colorlog_available,
    is_colorlog_available,
    raise_colorlog_missing_error,
)
from mlev.utils.imports.rich import (
    check_rich,
    is_rich_available,
    raise_rich_missing_error,
    rich_available,
)
from mlev.utils.imports.sklearn import (
    check_sklearn,
    is_sklearn_available,
    raise_sklearn_missing_error,
    sklearn_available,
)
