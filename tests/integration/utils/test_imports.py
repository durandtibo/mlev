from __future__ import annotations

import pytest

from mlev.testing.fixtures import (
    colorlog_available,
    colorlog_not_available,
    rich_available,
    rich_not_available,
    sklearn_available,
    sklearn_not_available,
)
from mlev.utils.imports import (
    check_colorlog,
    check_rich,
    check_sklearn,
    is_colorlog_available,
    is_rich_available,
    is_sklearn_available,
)

####################
#     colorlog     #
####################


@colorlog_available
def test_check_colorlog_with_package() -> None:
    check_colorlog()


@colorlog_not_available
def test_check_colorlog_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'colorlog' package is required but not installed."):
        check_colorlog()


@colorlog_available
def test_is_colorlog_available_true() -> None:
    assert is_colorlog_available()


@colorlog_not_available
def test_is_colorlog_available_false() -> None:
    assert not is_colorlog_available()


####################
#     rich     #
####################


@rich_available
def test_check_rich_with_package() -> None:
    check_rich()


@rich_not_available
def test_check_rich_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'rich' package is required but not installed."):
        check_rich()


@rich_available
def test_is_rich_available_true() -> None:
    assert is_rich_available()


@rich_not_available
def test_is_rich_available_false() -> None:
    assert not is_rich_available()


####################
#     sklearn     #
####################


@sklearn_available
def test_check_sklearn_with_package() -> None:
    check_sklearn()


@sklearn_not_available
def test_check_sklearn_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'sklearn' package is required but not installed."):
        check_sklearn()


@sklearn_available
def test_is_sklearn_available_true() -> None:
    assert is_sklearn_available()


@sklearn_not_available
def test_is_sklearn_available_false() -> None:
    assert not is_sklearn_available()
