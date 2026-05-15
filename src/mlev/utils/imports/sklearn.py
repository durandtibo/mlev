r"""Utilities to work with the optional ``scikit-learn`` dependency."""

from __future__ import annotations

__all__ = [
    "check_sklearn",
    "is_sklearn_available",
    "raise_sklearn_missing_error",
    "sklearn_available",
]

from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from coola.utils.imports import (
    decorator_package_available,
    package_available,
    raise_package_missing_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def check_sklearn() -> None:
    r"""Check if the ``sklearn`` package is installed.

    Raises:
        RuntimeError: if the ``sklearn`` package is not installed.

    Example:
        ```pycon
        >>> from mlev.utils.imports import check_sklearn
        >>> try:
        ...     check_sklearn()
        ... except RuntimeError:
        ...     pass

        ```
    """
    if not is_sklearn_available():
        raise_sklearn_missing_error()


@lru_cache
def is_sklearn_available() -> bool:
    r"""Indicate if the ``sklearn`` package is installed or not.

    Returns:
        ``True`` if ``sklearn`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from mlev.utils.imports import is_sklearn_available
        >>> is_sklearn_available()

        ```
    """
    return package_available("sklearn")


def sklearn_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``sklearn``
    is installed.

    Args:
        fn: The function to conditionally execute.

    Returns:
        A wrapper around ``fn``. When ``sklearn`` is unavailable, calling
            the wrapper returns ``None``.

    Example:
        ```pycon
        >>> from mlev.utils.imports import sklearn_available
        >>> @sklearn_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_sklearn_available)


def raise_sklearn_missing_error() -> NoReturn:
    r"""Raise a ``RuntimeError`` to indicate the ``sklearn`` package is
    missing.

    Raises:
        RuntimeError: Always, with a message indicating that the
            ``sklearn`` package is not installed.

    Example:
        ```pycon
        >>> from mlev.utils.imports import raise_sklearn_missing_error
        >>> try:
        ...     raise_sklearn_missing_error()
        ... except RuntimeError as e:
        ...     "'sklearn' package is required" in str(e)
        True

        ```
    """
    raise_package_missing_error("sklearn", "scikit-learn")
