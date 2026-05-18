from __future__ import annotations

import polars as pl
import pytest
from coola.equality import objects_are_equal

from mlev.utils.series import preprocess_pred

######################################
#     Tests for preprocess_pred      #
######################################


# --- drop_missing=False (default) ---


def test_preprocess_pred_no_missing_keep() -> None:
    y_true = pl.Series("y_true", [1, 0, 0, 1])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred)
    assert objects_are_equal(out_true, y_true)
    assert objects_are_equal(out_pred, y_pred)


def test_preprocess_pred_with_missing_keep() -> None:
    y_true = pl.Series("y_true", [1, 0, 0, 1, 1, None])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1, None, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred)
    assert objects_are_equal(out_true, y_true)
    assert objects_are_equal(out_pred, y_pred)


def test_preprocess_pred_returns_original_series_keep() -> None:
    y_true = pl.Series("y_true", [1, 0, 0, 1])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred)
    assert out_true is y_true
    assert out_pred is y_pred


# --- drop_missing=True ---


def test_preprocess_pred_no_missing_drop() -> None:
    y_true = pl.Series("y_true", [1, 0, 0, 1])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [1, 0, 0, 1]))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [0, 1, 0, 1]))


def test_preprocess_pred_missing_in_y_true_drop() -> None:
    y_true = pl.Series("y_true", [1, None, 0, 1])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [1, 0, 1]))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [0, 0, 1]))


def test_preprocess_pred_missing_in_y_pred_drop() -> None:
    y_true = pl.Series("y_true", [1, 0, 0, 1])
    y_pred = pl.Series("y_pred", [0, None, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [1, 0, 1]))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [0, 0, 1]))


def test_preprocess_pred_missing_in_both_drop() -> None:
    y_true = pl.Series("y_true", [1, 0, 0, 1, 1, None])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1, None, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [1, 0, 0, 1]))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [0, 1, 0, 1]))


def test_preprocess_pred_missing_overlap_drop() -> None:
    y_true = pl.Series("y_true", [1, None, 0, 1])
    y_pred = pl.Series("y_pred", [0, None, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [1, 0, 1]))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [0, 0, 1]))


def test_preprocess_pred_all_missing_drop() -> None:
    y_true = pl.Series("y_true", [None, None], dtype=pl.Int64)
    y_pred = pl.Series("y_pred", [None, None], dtype=pl.Int64)
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [], dtype=pl.Int64))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [], dtype=pl.Int64))


# --- Series names are preserved ---


def test_preprocess_pred_preserves_series_names_keep() -> None:
    y_true = pl.Series("y_true", [1, 0, 0, 1])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred)
    assert out_true.name == "y_true"
    assert out_pred.name == "y_pred"


def test_preprocess_pred_preserves_series_names_drop() -> None:
    y_true = pl.Series("y_true", [1, None, 0, 1])
    y_pred = pl.Series("y_pred", [0, 1, 0, 1])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert out_true.name == "y_true"
    assert out_pred.name == "y_pred"


# --- Different dtypes ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            pl.Series("y_true", [1, None, 0], dtype=pl.Int32),
            pl.Series("y_pred", [0, 1, None], dtype=pl.Int32),
            id="int32",
        ),
        pytest.param(
            pl.Series("y_true", [1.0, None, 0.0], dtype=pl.Float32),
            pl.Series("y_pred", [0.0, 1.0, None], dtype=pl.Float32),
            id="float32",
        ),
        pytest.param(
            pl.Series("y_true", [1.0, None, 0.0], dtype=pl.Float64),
            pl.Series("y_pred", [0.0, 1.0, None], dtype=pl.Float64),
            id="float64",
        ),
        pytest.param(
            pl.Series("y_true", ["a", None, "c"], dtype=pl.String),
            pl.Series("y_pred", ["d", "e", None], dtype=pl.String),
            id="str",
        ),
        pytest.param(
            pl.Series("y_true", [True, None, False], dtype=pl.Boolean),
            pl.Series("y_pred", [False, True, None], dtype=pl.Boolean),
            id="bool",
        ),
    ],
)
def test_preprocess_pred_drop_missing_dtypes(y_true: pl.Series, y_pred: pl.Series) -> None:
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert out_true.null_count() == 0
    assert out_pred.null_count() == 0
    assert len(out_true) == 1
    assert len(out_pred) == 1


# --- Shape mismatch ---


def test_preprocess_pred_different_shapes_raises() -> None:
    with pytest.raises(RuntimeError, match="series have different shapes:"):
        preprocess_pred(
            pl.Series("y_true", [1, 0, 0]),
            pl.Series("y_pred", [0, 1]),
        )


def test_preprocess_pred_different_shapes_drop_raises() -> None:
    with pytest.raises(RuntimeError, match="series have different shapes:"):
        preprocess_pred(
            pl.Series("y_true", [1, 0, 0]),
            pl.Series("y_pred", [0, 1]),
            drop_missing=True,
        )


# --- Edge cases ---


def test_preprocess_pred_empty_series() -> None:
    y_true = pl.Series("y_true", [], dtype=pl.Int64)
    y_pred = pl.Series("y_pred", [], dtype=pl.Int64)
    out_true, out_pred = preprocess_pred(y_true, y_pred)
    assert objects_are_equal(out_true, y_true)
    assert objects_are_equal(out_pred, y_pred)


def test_preprocess_pred_single_element_no_missing_drop() -> None:
    y_true = pl.Series("y_true", [1])
    y_pred = pl.Series("y_pred", [0])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [1]))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [0]))


def test_preprocess_pred_single_element_missing_drop() -> None:
    y_true = pl.Series("y_true", [None], dtype=pl.Int64)
    y_pred = pl.Series("y_pred", [1])
    out_true, out_pred = preprocess_pred(y_true, y_pred, drop_missing=True)
    assert objects_are_equal(out_true, pl.Series("y_true", [], dtype=pl.Int64))
    assert objects_are_equal(out_pred, pl.Series("y_pred", [], dtype=pl.Int64))
