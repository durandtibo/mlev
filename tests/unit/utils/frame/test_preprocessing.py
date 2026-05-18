from __future__ import annotations

import polars as pl
import pytest
from coola.equality import objects_are_equal

from mlev.utils.frame import preprocess

##################################
#     Tests for preprocess       #
##################################


# --- drop_missing=False (default) ---


def test_preprocess_no_missing_keep() -> None:
    frame = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert objects_are_equal(preprocess(frame), frame)


def test_preprocess_with_missing_keep() -> None:
    frame = pl.DataFrame({"x": [1, None, 3], "y": [4, 5, None]})
    assert objects_are_equal(preprocess(frame), frame)


def test_preprocess_returns_original_frame() -> None:
    # Verify the frame is returned as-is, not a copy
    frame = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert preprocess(frame) is frame


# --- drop_missing=True ---


def test_preprocess_no_missing_drop() -> None:
    frame = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert objects_are_equal(
        preprocess(frame, drop_missing=True),
        pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
    )


def test_preprocess_missing_in_first_column_drop() -> None:
    assert objects_are_equal(
        preprocess(
            pl.DataFrame({"x": [1, None, 3], "y": [4, 5, 6]}),
            drop_missing=True,
        ),
        pl.DataFrame({"x": [1, 3], "y": [4, 6]}),
    )


def test_preprocess_missing_in_last_column_drop() -> None:
    assert objects_are_equal(
        preprocess(
            pl.DataFrame({"x": [1, 2, 3], "y": [4, None, 6]}),
            drop_missing=True,
        ),
        pl.DataFrame({"x": [1, 3], "y": [4, 6]}),
    )


def test_preprocess_missing_in_multiple_columns_drop() -> None:
    assert objects_are_equal(
        preprocess(
            pl.DataFrame({"x": [1, None, 3], "y": [None, 5, 6]}),
            drop_missing=True,
        ),
        pl.DataFrame({"x": [3], "y": [6]}),
    )


def test_preprocess_missing_in_same_row_drop() -> None:
    # Both columns have null in the same row — only one row dropped
    assert objects_are_equal(
        preprocess(
            pl.DataFrame({"x": [1, None, 3], "y": [4, None, 6]}),
            drop_missing=True,
        ),
        pl.DataFrame({"x": [1, 3], "y": [4, 6]}),
    )


def test_preprocess_all_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            pl.DataFrame(
                {"x": [None, None], "y": [None, None]},
                schema={"x": pl.Int64, "y": pl.Int64},
            ),
            drop_missing=True,
        ),
        pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Int64, "y": pl.Int64}),
    )


def test_preprocess_y_true_y_pred_drop() -> None:
    # Mirrors the docstring example
    assert objects_are_equal(
        preprocess(
            pl.DataFrame({"y_true": [1, 0, 0, 1, 1, None], "y_pred": [0, 1, 0, 1, None, 1]}),
            drop_missing=True,
        ),
        pl.DataFrame({"y_true": [1, 0, 0, 1], "y_pred": [0, 1, 0, 1]}),
    )


# --- Column names and schema are preserved ---


def test_preprocess_preserves_column_names_keep() -> None:
    frame = pl.DataFrame({"y_true": [1, 2], "y_pred": [3, 4]})
    assert preprocess(frame).columns == ["y_true", "y_pred"]


def test_preprocess_preserves_column_names_drop() -> None:
    frame = pl.DataFrame({"y_true": [1, None], "y_pred": [3, 4]})
    assert preprocess(frame, drop_missing=True).columns == ["y_true", "y_pred"]


def test_preprocess_preserves_schema_drop() -> None:
    frame = pl.DataFrame(
        {"x": [1, None, 3], "y": [4.0, 5.0, None]},
        schema={"x": pl.Int32, "y": pl.Float32},
    )
    result = preprocess(frame, drop_missing=True)
    assert result.schema == frame.schema


# --- Different dtypes ---


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int8}),
            pl.DataFrame({"x": [1, 3]}, schema={"x": pl.Int8}),
            id="int8",
        ),
        pytest.param(
            pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int16}),
            pl.DataFrame({"x": [1, 3]}, schema={"x": pl.Int16}),
            id="int16",
        ),
        pytest.param(
            pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int32}),
            pl.DataFrame({"x": [1, 3]}, schema={"x": pl.Int32}),
            id="int32",
        ),
        pytest.param(
            pl.DataFrame({"x": [1, None, 3]}, schema={"x": pl.Int64}),
            pl.DataFrame({"x": [1, 3]}, schema={"x": pl.Int64}),
            id="int64",
        ),
        pytest.param(
            pl.DataFrame({"x": [1.0, None, 3.0]}, schema={"x": pl.Float32}),
            pl.DataFrame({"x": [1.0, 3.0]}, schema={"x": pl.Float32}),
            id="float32",
        ),
        pytest.param(
            pl.DataFrame({"x": [1.0, None, 3.0]}, schema={"x": pl.Float64}),
            pl.DataFrame({"x": [1.0, 3.0]}, schema={"x": pl.Float64}),
            id="float64",
        ),
        pytest.param(
            pl.DataFrame({"x": [True, None, False]}, schema={"x": pl.Boolean}),
            pl.DataFrame({"x": [True, False]}, schema={"x": pl.Boolean}),
            id="bool",
        ),
        pytest.param(
            pl.DataFrame({"x": ["a", None, "c"]}, schema={"x": pl.String}),
            pl.DataFrame({"x": ["a", "c"]}, schema={"x": pl.String}),
            id="str",
        ),
        pytest.param(
            pl.DataFrame({"x": ["2021-01-01", None, "2021-01-03"]}, schema={"x": pl.Date}),
            pl.DataFrame({"x": ["2021-01-01", "2021-01-03"]}, schema={"x": pl.Date}),
            id="date",
        ),
    ],
)
def test_preprocess_drop_dtypes(frame: pl.DataFrame, expected: pl.DataFrame) -> None:
    assert objects_are_equal(preprocess(frame, drop_missing=True), expected)


# --- NaN is not null ---


def test_preprocess_nan_is_not_missing_keep() -> None:
    frame = pl.DataFrame({"x": [1.0, float("nan"), 3.0]})
    assert objects_are_equal(preprocess(frame), frame)


def test_preprocess_nan_is_not_missing_drop() -> None:
    # NaN rows should NOT be dropped since NaN is not null in Polars
    frame = pl.DataFrame({"x": [1.0, float("nan"), 3.0]})
    assert objects_are_equal(preprocess(frame, drop_missing=True), frame, equal_nan=True)


# --- Edge cases ---


def test_preprocess_empty_frame_keep() -> None:
    frame = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Int64, "y": pl.Int64})
    assert objects_are_equal(preprocess(frame), frame)


def test_preprocess_empty_frame_drop() -> None:
    frame = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Int64, "y": pl.Int64})
    assert objects_are_equal(preprocess(frame, drop_missing=True), frame)


def test_preprocess_single_row_no_missing_drop() -> None:
    frame = pl.DataFrame({"x": [1], "y": [2]})
    assert objects_are_equal(
        preprocess(frame, drop_missing=True),
        pl.DataFrame({"x": [1], "y": [2]}),
    )


def test_preprocess_single_row_with_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            pl.DataFrame({"x": [None], "y": [1]}, schema={"x": pl.Int64, "y": pl.Int64}),
            drop_missing=True,
        ),
        pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Int64, "y": pl.Int64}),
    )


def test_preprocess_single_column_drop() -> None:
    assert objects_are_equal(
        preprocess(pl.DataFrame({"x": [1, None, 3]}), drop_missing=True),
        pl.DataFrame({"x": [1, 3]}),
    )


def test_preprocess_many_columns_one_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            pl.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": [4, 5, 6],
                    "c": [7, None, 9],
                    "d": [10, 11, 12],
                }
            ),
            drop_missing=True,
        ),
        pl.DataFrame(
            {
                "a": [1, 3],
                "b": [4, 6],
                "c": [7, 9],
                "d": [10, 12],
            }
        ),
    )
