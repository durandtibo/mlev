r"""Contain utility functions to check and import optional dependencies.

This package exposes helper functions and decorators for the optional
``colorlog`` and ``rich`` packages, such as
:func:`is_colorlog_available`, :func:`is_rich_available`, and the
:func:`colorlog_available` / :func:`rich_available` decorators.
"""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "check_rich",
    "colorlog_available",
    "is_colorlog_available",
    "is_rich_available",
    "raise_colorlog_missing_error",
    "raise_rich_missing_error",
    "rich_available",
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
