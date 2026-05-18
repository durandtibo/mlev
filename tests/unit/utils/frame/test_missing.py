from __future__ import annotations

import polars as pl
import pytest

from mlev.utils.frame import contains_missing
from mlev.utils.missing import MISSING_POLICIES, MissingPolicy

##################################
#   Tests for contains_missing   #
##################################


# --- No missing values ---


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_contains_missing_frame_no_missing(missing_policy: MissingPolicy) -> None:
    assert not contains_missing(
        pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        missing_policy=missing_policy,
    )


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_contains_missing_frame_single_column_no_missing(missing_policy: MissingPolicy) -> None:
    assert not contains_missing(
        pl.DataFrame({"x": [1, 2, 3]}),
        missing_policy=missing_policy,
    )


# --- Missing values present ---


def test_contains_missing_frame_missing_in_first_column() -> None:
    assert contains_missing(pl.DataFrame({"x": [1, None, 3], "y": [4, 5, 6]}))


def test_contains_missing_frame_missing_in_last_column() -> None:
    assert contains_missing(pl.DataFrame({"x": [1, 2, 3], "y": [4, None, 6]}))


def test_contains_missing_frame_missing_in_all_columns() -> None:
    assert contains_missing(pl.DataFrame({"x": [1, None, 3], "y": [None, 5, 6]}))


def test_contains_missing_frame_all_values_missing() -> None:
    assert contains_missing(
        pl.DataFrame({"x": [None, None], "y": [None, None]}, schema={"x": pl.Int64, "y": pl.Int64})
    )


def test_contains_missing_frame_single_column_with_missing() -> None:
    assert contains_missing(pl.DataFrame({"x": [1, None, 3]}))


def test_contains_missing_frame_single_missing_value() -> None:
    assert contains_missing(
        pl.DataFrame({"x": [None], "y": [1]}, schema={"x": pl.Int64, "y": pl.Int64})
    )


# --- NaN is not missing ---


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_contains_missing_frame_nan_is_not_missing(missing_policy: MissingPolicy) -> None:
    assert not contains_missing(
        pl.DataFrame({"x": [1.0, float("nan"), 3.0], "y": [4.0, 5.0, 6.0]}),
        missing_policy=missing_policy,
    )


# --- missing_policy='propagate' ---


def test_contains_missing_frame_propagate_returns_true() -> None:
    assert contains_missing(
        pl.DataFrame({"x": [1, None, 3]}),
        missing_policy="propagate",
    )


def test_contains_missing_frame_propagate_returns_false() -> None:
    assert not contains_missing(
        pl.DataFrame({"x": [1, 2, 3]}),
        missing_policy="propagate",
    )


def test_contains_missing_frame_propagate_does_not_raise() -> None:
    # 'propagate' should never raise even with missing values
    result = contains_missing(
        pl.DataFrame({"x": [1, None, 3]}),
        missing_policy="propagate",
    )
    assert result is True


# --- missing_policy='omit' ---


def test_contains_missing_frame_omit_returns_true() -> None:
    assert contains_missing(
        pl.DataFrame({"x": [1, None, 3]}),
        missing_policy="omit",
    )


def test_contains_missing_frame_omit_returns_false() -> None:
    assert not contains_missing(
        pl.DataFrame({"x": [1, 2, 3]}),
        missing_policy="omit",
    )


def test_contains_missing_frame_omit_does_not_raise() -> None:
    # 'omit' should never raise even with missing values
    result = contains_missing(
        pl.DataFrame({"x": [1, None, 3]}),
        missing_policy="omit",
    )
    assert result is True


# --- missing_policy='raise' ---


def test_contains_missing_frame_raise_no_missing_does_not_raise() -> None:
    assert not contains_missing(
        pl.DataFrame({"x": [1, 2, 3]}),
        missing_policy="raise",
    )


def test_contains_missing_frame_raise_with_missing_raises() -> None:
    with pytest.raises(ValueError, match="input contains at least one missing value"):
        contains_missing(
            pl.DataFrame({"x": [1, None, 3]}),
            missing_policy="raise",
        )


def test_contains_missing_frame_raise_custom_name() -> None:
    with pytest.raises(ValueError, match="my_frame contains at least one missing value"):
        contains_missing(
            pl.DataFrame({"x": [1, None, 3]}),
            missing_policy="raise",
            name="my_frame",
        )


def test_contains_missing_frame_raise_missing_in_any_column() -> None:
    with pytest.raises(ValueError, match="input contains at least one missing value"):
        contains_missing(
            pl.DataFrame({"x": [1, 2, 3], "y": [4, None, 6]}),
            missing_policy="raise",
        )


# --- Different dtypes ---


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int8}), id="int8"),
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int16}), id="int16"),
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int32}), id="int32"),
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int64}), id="int64"),
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.UInt8}), id="uint8"),
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.UInt16}), id="uint16"),
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.UInt32}), id="uint32"),
        pytest.param(pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.UInt64}), id="uint64"),
        pytest.param(pl.DataFrame({"x": [1.0, None, 3.0]}, schema={"x": pl.Float32}), id="float32"),
        pytest.param(pl.DataFrame({"x": [1.0, None, 3.0]}, schema={"x": pl.Float64}), id="float64"),
        pytest.param(pl.DataFrame({"x": [True, None, False]}, schema={"x": pl.Boolean}), id="bool"),
        pytest.param(pl.DataFrame({"x": ["a", None, "c"]}, schema={"x": pl.String}), id="str"),
        pytest.param(
            pl.DataFrame({"x": ["2021-01-01", None, "2021-01-03"]}, schema={"x": pl.Date}),
            id="date",
        ),
    ],
)
def test_contains_missing_frame_dtypes_with_missing(frame: pl.DataFrame) -> None:
    assert contains_missing(frame)


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.Int8}), id="int8"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.Int16}), id="int16"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.Int32}), id="int32"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.Int64}), id="int64"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.UInt8}), id="uint8"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.UInt16}), id="uint16"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.UInt32}), id="uint32"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.UInt64}), id="uint64"),
        pytest.param(pl.DataFrame({"x": [1.0, 2.0, 3.0]}, schema={"x": pl.Float32}), id="float32"),
        pytest.param(pl.DataFrame({"x": [1.0, 2.0, 3.0]}, schema={"x": pl.Float64}), id="float64"),
        pytest.param(pl.DataFrame({"x": [True, False, True]}, schema={"x": pl.Boolean}), id="bool"),
        pytest.param(pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": pl.String}), id="str"),
        pytest.param(
            pl.DataFrame({"x": ["2021-01-01", "2021-01-02", "2021-01-03"]}, schema={"x": pl.Date}),
            id="date",
        ),
    ],
)
def test_contains_missing_frame_dtypes_no_missing(frame: pl.DataFrame) -> None:
    assert not contains_missing(frame)


# --- Mixed dtypes ---


def test_contains_missing_frame_mixed_dtypes_with_missing() -> None:
    assert contains_missing(
        pl.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.0, None, 3.0],
                "str": ["a", "b", "c"],
            }
        )
    )


def test_contains_missing_frame_mixed_dtypes_no_missing() -> None:
    assert not contains_missing(
        pl.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.0, 2.0, 3.0],
                "str": ["a", "b", "c"],
            }
        )
    )


# --- Edge cases ---


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_contains_missing_frame_empty_rows(missing_policy: MissingPolicy) -> None:
    assert not contains_missing(
        pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Int64, "y": pl.Int64}),
        missing_policy=missing_policy,
    )


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_contains_missing_frame_empty_no_columns(missing_policy: MissingPolicy) -> None:
    assert not contains_missing(
        pl.DataFrame(),
        missing_policy=missing_policy,
    )


def test_contains_missing_frame_single_row_no_missing() -> None:
    assert not contains_missing(pl.DataFrame({"x": [1], "y": [2]}))


def test_contains_missing_frame_single_row_with_missing() -> None:
    assert contains_missing(
        pl.DataFrame({"x": [None], "y": [2]}, schema={"x": pl.Int64, "y": pl.Int64})
    )


def test_contains_missing_frame_many_columns_one_missing() -> None:
    assert contains_missing(
        pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
                "d": [10, None, 12],
                "e": [13, 14, 15],
            }
        )
    )
