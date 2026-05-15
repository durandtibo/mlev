from __future__ import annotations

import polars as pl
import pytest

from mlev.utils.missing import MISSING_POLICIES
from mlev.utils.series import contains_missing, is_missing, multi_is_missing

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


################################
#     Tests for is_missing     #
################################


# --- Basic cases ---


def test_is_missing_no_null() -> None:
    assert is_missing(pl.Series("x", [1, 2, 3])).equals(
        pl.Series("is_missing", [False, False, False])
    )


def test_is_missing_with_null() -> None:
    assert is_missing(pl.Series("x", [1, None, 3])).equals(
        pl.Series("is_missing", [False, True, False])
    )


def test_is_missing_all_null() -> None:
    assert is_missing(pl.Series("x", [None, None, None], dtype=pl.Int64)).equals(
        pl.Series("is_missing", [True, True, True])
    )


def test_is_missing_null_at_start() -> None:
    assert is_missing(pl.Series("x", [None, 2, 3])).equals(
        pl.Series("is_missing", [True, False, False])
    )


def test_is_missing_null_at_end() -> None:
    assert is_missing(pl.Series("x", [1, 2, None])).equals(
        pl.Series("is_missing", [False, False, True])
    )


# --- Output series name ---


def test_is_missing_default_name() -> None:
    assert is_missing(pl.Series("x", [1, 2, 3])).name == "is_missing"


def test_is_missing_custom_name() -> None:
    assert is_missing(pl.Series("x", [1, 2, 3]), name="mask").name == "mask"


# --- NaN is not null ---


def test_is_missing_nan_is_not_missing() -> None:
    assert is_missing(pl.Series("x", [1.0, float("nan"), 3.0])).equals(
        pl.Series("is_missing", [False, False, False])
    )


# --- Different dtypes ---


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int8), id="int8"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int16), id="int16"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int32), id="int32"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int64), id="int64"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt8), id="uint8"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt16), id="uint16"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt32), id="uint32"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt64), id="uint64"),
        pytest.param(pl.Series("x", [1.0, None, 3.0], dtype=pl.Float32), id="float32"),
        pytest.param(pl.Series("x", [1.0, None, 3.0], dtype=pl.Float64), id="float64"),
        pytest.param(pl.Series("x", [True, None, False], dtype=pl.Boolean), id="bool"),
        pytest.param(pl.Series("x", ["a", None, "c"], dtype=pl.String), id="str"),
        pytest.param(pl.Series("x", ["2021-01-01", None, "2021-01-03"], dtype=pl.Date), id="date"),
    ],
)
def test_is_missing_dtypes_with_null(series: pl.Series) -> None:
    assert is_missing(series).equals(pl.Series("is_missing", [False, True, False]))


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.Int8), id="int8"),
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.Int16), id="int16"),
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.Int32), id="int32"),
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.Int64), id="int64"),
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.UInt8), id="uint8"),
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.UInt16), id="uint16"),
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.UInt32), id="uint32"),
        pytest.param(pl.Series("x", [1, 2, 3], dtype=pl.UInt64), id="uint64"),
        pytest.param(pl.Series("x", [1.0, 2.0, 3.0], dtype=pl.Float32), id="float32"),
        pytest.param(pl.Series("x", [1.0, 2.0, 3.0], dtype=pl.Float64), id="float64"),
        pytest.param(pl.Series("x", [True, False, True], dtype=pl.Boolean), id="bool"),
        pytest.param(pl.Series("x", ["a", "b", "c"], dtype=pl.String), id="str"),
        pytest.param(
            pl.Series("x", ["2021-01-01", "2021-01-02", "2021-01-03"], dtype=pl.Date), id="date"
        ),
    ],
)
def test_is_missing_dtypes_no_null(series: pl.Series) -> None:
    assert is_missing(series).equals(pl.Series("is_missing", [False, False, False]))


# --- Edge cases ---


def test_is_missing_empty_series_raises() -> None:
    assert is_missing(pl.Series("x", [], dtype=pl.Int64)).equals(
        pl.Series("is_missing", [], dtype=pl.Boolean)
    )


def test_is_missing_single_element_null() -> None:
    assert is_missing(pl.Series("x", [None], dtype=pl.Int64)).equals(
        pl.Series("is_missing", [True])
    )


def test_is_missing_single_element_no_null() -> None:
    assert is_missing(pl.Series("x", [1])).equals(pl.Series("is_missing", [False]))


######################################
#     Tests for multi_is_missing     #
######################################


# --- Single series ---


def test_multi_is_missing_single_series_no_null() -> None:
    assert multi_is_missing([pl.Series("x", [1, 2, 3])]).equals(
        pl.Series("is_missing", [False, False, False])
    )


def test_multi_is_missing_single_series_with_null() -> None:
    assert multi_is_missing([pl.Series("x", [1, None, 3])]).equals(
        pl.Series("is_missing", [False, True, False])
    )


def test_multi_is_missing_single_series_all_null() -> None:
    assert multi_is_missing([pl.Series("x", [None, None, None], dtype=pl.Int64)]).equals(
        pl.Series("is_missing", [True, True, True])
    )


# --- Multiple series ---


def test_multi_is_missing_two_series_no_null() -> None:
    assert multi_is_missing([pl.Series("x", [1, 2, 3]), pl.Series("y", [4, 5, 6])]).equals(
        pl.Series("is_missing", [False, False, False])
    )


def test_multi_is_missing_two_series_first_has_null() -> None:
    assert multi_is_missing([pl.Series("x", [1, None, 3]), pl.Series("y", [4, 5, 6])]).equals(
        pl.Series("is_missing", [False, True, False])
    )


def test_multi_is_missing_two_series_second_has_null() -> None:
    assert multi_is_missing([pl.Series("x", [1, 2, 3]), pl.Series("y", [4, None, 6])]).equals(
        pl.Series("is_missing", [False, True, False])
    )


def test_multi_is_missing_two_series_both_have_null() -> None:
    assert multi_is_missing([pl.Series("x", [1, None, 3]), pl.Series("y", [None, 5, 6])]).equals(
        pl.Series("is_missing", [True, True, False])
    )


def test_multi_is_missing_two_series_null_overlap() -> None:
    # Both series have null at the same position
    assert multi_is_missing([pl.Series("x", [1, None, 3]), pl.Series("y", [4, None, 6])]).equals(
        pl.Series("is_missing", [False, True, False])
    )


def test_multi_is_missing_three_series() -> None:
    assert multi_is_missing(
        [
            pl.Series("x", [1, None, 3]),
            pl.Series("y", [4, 5, None]),
            pl.Series("z", [None, 8, 9]),
        ]
    ).equals(pl.Series("is_missing", [True, True, True]))


# --- Output series name ---


def test_multi_is_missing_default_name() -> None:
    result = multi_is_missing([pl.Series("x", [1, 2, 3])])
    assert result.name == "is_missing"


def test_multi_is_missing_custom_name() -> None:
    result = multi_is_missing([pl.Series("x", [1, 2, 3])], name="mask")
    assert result.name == "mask"


# --- NaN is not null ---


def test_multi_is_missing_nan_is_not_null() -> None:
    assert multi_is_missing([pl.Series("x", [1.0, float("nan"), 3.0])]).equals(
        pl.Series("is_missing", [False, False, False])
    )


# --- Different dtypes ---


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int8), id="int8"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int16), id="int16"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int32), id="int32"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.Int64), id="int64"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt8), id="uint8"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt16), id="uint16"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt32), id="uint32"),
        pytest.param(pl.Series("x", [1, None, 3], dtype=pl.UInt64), id="uint64"),
        pytest.param(pl.Series("x", [1.0, None, 3.0], dtype=pl.Float32), id="float32"),
        pytest.param(pl.Series("x", [1.0, None, 3.0], dtype=pl.Float64), id="float64"),
        pytest.param(pl.Series("x", [True, None, False], dtype=pl.Boolean), id="bool"),
        pytest.param(pl.Series("x", ["a", None, "c"], dtype=pl.String), id="str"),
        pytest.param(pl.Series("x", ["2021-01-01", None, "2021-01-03"], dtype=pl.Date), id="date"),
    ],
)
def test_multi_is_missing_dtypes_with_null(series: pl.Series) -> None:
    assert multi_is_missing([series]).equals(pl.Series("is_missing", [False, True, False]))


# --- Edge cases ---


def test_multi_is_missing_empty_series_raises() -> None:
    with pytest.raises(ValueError, match="'series' cannot be empty"):
        multi_is_missing([])


def test_multi_is_missing_single_element_null() -> None:
    assert multi_is_missing([pl.Series("x", [None], dtype=pl.Int64)]).equals(
        pl.Series("is_missing", [True])
    )


def test_multi_is_missing_single_element_no_null() -> None:
    assert multi_is_missing([pl.Series("x", [1])]).equals(pl.Series("is_missing", [False]))
