from __future__ import annotations

import polars as pl
import pytest

from mlev.utils.series import check_same_shape

######################################
#     Tests for check_same_shape     #
######################################


def test_check_same_shape_1_array() -> None:
    check_same_shape([pl.Series("col", [1, 0, 0, 1, 1])])


def test_check_same_shape_2_series_correct() -> None:
    check_same_shape([pl.Series("col", [1, 0, 0, 1, 1]), pl.Series("col", [1, 2, 3, 4, 5])])


def test_check_same_shape_2_series_incorrect() -> None:
    with pytest.raises(RuntimeError, match="series have different shapes"):
        check_same_shape([pl.Series("col", [1, 0, 0, 1, 1]), pl.Series("col", [1, 0, 0, 1])])


def test_check_same_shape_3_series_correct() -> None:
    check_same_shape(
        [
            pl.Series("col", [1, 0, 0, 1, 1]),
            pl.Series("col", [1, 2, 3, 4, 5]),
            pl.Series("col", [5, 4, 3, 2, 1]),
        ]
    )


def test_check_same_shape_3_series_incorrect() -> None:
    with pytest.raises(RuntimeError, match="series have different shapes"):
        check_same_shape(
            [
                pl.Series("col", [1, 0, 0, 1, 1]),
                pl.Series("col", [1, 2, 3, 4]),
                pl.Series("col", [6, 5, 4, 3, 2, 1]),
            ]
        )
