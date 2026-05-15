from __future__ import annotations

import polars as pl
import pytest

from mlev.utils.missing import MISSING_POLICIES
from mlev.utils.series import contains_missing

######################################
#     Tests for contains_missing     #
######################################


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.Int8), id="int8"),
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.Int16), id="int16"),
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.Int32), id="int32"),
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.Int64), id="int64"),
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.UInt8), id="uint8"),
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.UInt16), id="uint16"),
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.UInt32), id="uint32"),
        pytest.param(pl.Series("col", [1, 2, 3], dtype=pl.UInt64), id="uint64"),
        pytest.param(pl.Series("col", [1.0, 2.0, 3.0], dtype=pl.Float32), id="float32"),
        pytest.param(pl.Series("col", [1.0, 2.0, 3.0], dtype=pl.Float64), id="float64"),
        pytest.param(pl.Series("col", [True, False, True], dtype=pl.Boolean), id="bool"),
        pytest.param(pl.Series("col", ["a", "b", "c"], dtype=pl.String), id="str"),
        pytest.param(
            pl.Series("col", ["2021-01-01", "2021-01-02", "2021-01-03"], dtype=pl.Date), id="date"
        ),
    ],
)
def test_contains_missing_no_missing(series: pl.Series, missing_policy: str) -> None:
    assert not contains_missing(series, missing_policy=missing_policy)


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.Int8), id="int8"),
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.Int16), id="int16"),
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.Int32), id="int32"),
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.Int64), id="int64"),
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.UInt8), id="uint8"),
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.UInt16), id="uint16"),
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.UInt32), id="uint32"),
        pytest.param(pl.Series("col", [1, None, 3], dtype=pl.UInt64), id="uint64"),
        pytest.param(pl.Series("col", [1.0, None, 3.0], dtype=pl.Float32), id="float32"),
        pytest.param(pl.Series("col", [1.0, None, 3.0], dtype=pl.Float64), id="float64"),
        pytest.param(pl.Series("col", [True, None, True], dtype=pl.Boolean), id="bool"),
        pytest.param(pl.Series("col", ["a", None, "c"], dtype=pl.String), id="str"),
        pytest.param(
            pl.Series("col", ["2021-01-01", None, "2021-01-03"], dtype=pl.Date), id="date"
        ),
    ],
)
def test_contains_missing_with_missing(series: pl.Series) -> None:
    assert contains_missing(series)


def test_contains_missing_raise() -> None:
    with pytest.raises(ValueError, match="col contains at least one missing value"):
        contains_missing(pl.Series("col", [1, 2, None, 4]), missing_policy="raise")


def test_contains_missing_raise_series_name() -> None:
    with pytest.raises(ValueError, match="my_series contains at least one missing value"):
        contains_missing(pl.Series("my_series", [1, 2, None, 4]), missing_policy="raise")


# --- NaN is not null in Polars ---


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("col", [1.0, float("nan"), 3.0], dtype=pl.Float32), id="float32"),
        pytest.param(pl.Series("col", [1.0, float("nan"), 3.0], dtype=pl.Float64), id="float64"),
    ],
)
def test_contains_missing_nan_is_not_missing(series: pl.Series, missing_policy: str) -> None:
    assert not contains_missing(series, missing_policy=missing_policy)


# --- Edge cases ---


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("col", [], dtype=pl.Int64), id="empty_int64"),
        pytest.param(pl.Series("col", [], dtype=pl.Float64), id="empty_float64"),
        pytest.param(pl.Series("col", [], dtype=pl.String), id="empty_str"),
    ],
)
def test_contains_missing_empty(series: pl.Series) -> None:
    assert not contains_missing(series)


def test_contains_missing_all_none() -> None:
    assert contains_missing(pl.Series("col", [None, None, None], dtype=pl.Int64))


def test_contains_missing_single_none() -> None:
    assert contains_missing(pl.Series("col", [None], dtype=pl.Int64))


def test_contains_missing_single_value() -> None:
    assert not contains_missing(pl.Series("col", [1]))
