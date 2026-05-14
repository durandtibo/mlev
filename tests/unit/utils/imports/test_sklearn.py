from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from mlev.utils.imports import (
    check_sklearn,
    is_sklearn_available,
    raise_sklearn_missing_error,
    sklearn_available,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_sklearn_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


###################
#     sklearn     #
###################


def test_check_sklearn_with_package() -> None:
    with patch("mlev.utils.imports.sklearn.is_sklearn_available", lambda: True):
        check_sklearn()


def test_check_sklearn_without_package() -> None:
    with (
        patch("mlev.utils.imports.sklearn.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'scikit-learn' package is required but not installed."),
    ):
        check_sklearn()


def test_is_sklearn_available() -> None:
    assert isinstance(is_sklearn_available(), bool)


def test_sklearn_available_with_package() -> None:
    with patch("mlev.utils.imports.sklearn.is_sklearn_available", lambda: True):
        fn = sklearn_available(my_function)
        assert fn(2) == 44


def test_sklearn_available_without_package() -> None:
    with patch("mlev.utils.imports.sklearn.is_sklearn_available", lambda: False):
        fn = sklearn_available(my_function)
        assert fn(2) is None


def test_sklearn_available_decorator_with_package() -> None:
    with patch("mlev.utils.imports.sklearn.is_sklearn_available", lambda: True):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_sklearn_available_decorator_without_package() -> None:
    with patch("mlev.utils.imports.sklearn.is_sklearn_available", lambda: False):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_sklearn_missing_error() -> None:
    with pytest.raises(
        RuntimeError, match=r"'scikit-learn' package is required but not installed."
    ):
        raise_sklearn_missing_error()
