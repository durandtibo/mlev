from __future__ import annotations

import polars as pl
import pytest

from mlev.functional.frame import accuracy
from mlev.results import AccuracyResult

##############################
#     Tests for accuracy     #
##############################


# ----------------------------------------------------
# Correct predictions
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1, 1], "y_pred": [1, 0, 0, 1, 1]}),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="int64-all-correct",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1, 1], "y_pred": [0, 1, 1, 0, 0]}),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="int64-all-incorrect",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1], "y_pred": [1, 1, 0, 1]}),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="int64-partial",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": [1.0, 0.0, 0.0, 1.0, 1.0], "y_pred": [1.0, 0.0, 0.0, 1.0, 1.0]}
            ),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="float64-all-correct",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": [1.0, 0.0, 0.0, 1.0, 1.0], "y_pred": [0.0, 1.0, 1.0, 0.0, 0.0]}
            ),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="float64-all-incorrect",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1.0, 0.0, 0.0, 1.0], "y_pred": [1.0, 1.0, 0.0, 1.0]}),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="float64-partial",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": ["cat", "dog", "cat", "dog"], "y_pred": ["cat", "dog", "cat", "dog"]}
            ),
            AccuracyResult(num_correct_predictions=4, num_predictions=4),
            id="str-all-correct",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": ["cat", "dog", "cat", "dog"], "y_pred": ["dog", "cat", "dog", "cat"]}
            ),
            AccuracyResult(num_correct_predictions=0, num_predictions=4),
            id="str-all-incorrect",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": ["cat", "dog", "cat", "dog"], "y_pred": ["cat", "cat", "cat", "dog"]}
            ),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="str-partial",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": ["cat", "dog", "bird", "fish"], "y_pred": ["cat", "dog", "fish", "fish"]}
            ),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="str-multiclass-partial",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [True, False, True], "y_pred": [True, False, True]}),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="bool-all-correct",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [True, False, True], "y_pred": [False, True, False]}),
            AccuracyResult(num_correct_predictions=0, num_predictions=3),
            id="bool-all-incorrect",
        ),
    ],
)
def test_accuracy_frame(frame: pl.DataFrame, expected: AccuracyResult) -> None:
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred").equal(expected)


# ----------------------------------------------------
# missing_policy='propagate' (default)
# ----------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, None, 1], "y_pred": [1, 0, 0, 1]}),
            id="int64-null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1], "y_pred": [1, 0, None, 1]}),
            id="int64-null-in-y_pred",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, None, 0, 1], "y_pred": [1, None, None, 1]}),
            id="int64-null-in-both",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1.0, 0.0, None, 1.0], "y_pred": [1.0, 0.0, 0.0, 1.0]}),
            id="float64-null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat", None, "cat"], "y_pred": ["cat", "dog", "cat"]}),
            id="str-null-in-y_true",
        ),
    ],
)
def test_accuracy_frame_missing_propagate(frame: pl.DataFrame) -> None:
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=len(frame)),
        equal_nan=True,
    )


# ----------------------------------------------------
# missing_policy='omit'
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, None, 1], "y_pred": [1, 0, 0, 1]}),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="int64-null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1], "y_pred": [1, 0, None, 1]}),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="int64-null-in-y_pred",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, None, 1, 1, None], "y_pred": [1, None, 0, 1, 0, None]}),
            AccuracyResult(num_correct_predictions=2, num_predictions=3),
            id="int64-null-in-both",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": [None, None], "y_pred": [None, None]},
                schema={"y_true": pl.Int64, "y_pred": pl.Int64},
            ),
            AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
            id="int64-all-null",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1.0, None, 0.0, 1.0], "y_pred": [1.0, 0.0, 0.0, 1.0]}),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="float64-null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat", None, "cat"], "y_pred": ["cat", "dog", "dog"]}),
            AccuracyResult(num_correct_predictions=1, num_predictions=2),
            id="str-null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat", "dog", "cat"], "y_pred": ["cat", None, "dog"]}),
            AccuracyResult(num_correct_predictions=1, num_predictions=2),
            id="str-null-in-y_pred",
        ),
    ],
)
def test_accuracy_frame_missing_omit(frame: pl.DataFrame, expected: AccuracyResult) -> None:
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="omit").equal(
        expected, equal_nan=True
    )


# ----------------------------------------------------
# missing_policy='raise'
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1], "y_pred": [1, 0, 0, 1]}),
            AccuracyResult(num_correct_predictions=4, num_predictions=4),
            id="int64-no-missing",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat", "dog"], "y_pred": ["cat", "dog"]}),
            AccuracyResult(num_correct_predictions=2, num_predictions=2),
            id="str-no-missing",
        ),
    ],
)
def test_accuracy_frame_no_missing_raise(frame: pl.DataFrame, expected: AccuracyResult) -> None:
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="raise").equal(
        expected
    )


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, None, 0], "y_pred": [1, 0, 0]}),
            id="int64-null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0], "y_pred": [1, None, 0]}),
            id="int64-null-in-y_pred",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat", None, "dog"], "y_pred": ["cat", "dog", "dog"]}),
            id="str-null-in-y_true",
        ),
    ],
)
def test_accuracy_frame_missing_raise(frame: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="input contains at least one missing value"):
        accuracy(frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="raise")


# ----------------------------------------------------
# Extra columns are ignored
# ----------------------------------------------------


def test_accuracy_frame_extra_columns_ignored() -> None:
    frame = pl.DataFrame(
        {
            "y_true": [1, 0, 0, 1],
            "y_pred": [1, 0, 0, 1],
            "extra": [10, 20, 30, 40],
        }
    )
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        AccuracyResult(num_correct_predictions=4, num_predictions=4)
    )


def test_accuracy_frame_extra_column_with_missing_ignored() -> None:
    # Missing values in extra columns should not affect the result
    frame = pl.DataFrame(
        {
            "y_true": [1, 0, 0, 1],
            "y_pred": [1, 0, 0, 1],
            "extra": [10, None, 30, 40],
        }
    )
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        AccuracyResult(num_correct_predictions=4, num_predictions=4)
    )


# ----------------------------------------------------
# Edge cases
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame(
                {"y_true": [], "y_pred": []}, schema={"y_true": pl.Int64, "y_pred": pl.Int64}
            ),
            AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
            id="empty-int64",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": [], "y_pred": []}, schema={"y_true": pl.String, "y_pred": pl.String}
            ),
            AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
            id="empty-str",
        ),
    ],
)
def test_accuracy_frame_empty(frame: pl.DataFrame, expected: AccuracyResult) -> None:
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred").equal(expected, equal_nan=True)


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame({"y_true": [1], "y_pred": [1]}),
            AccuracyResult(num_correct_predictions=1, num_predictions=1),
            id="single-correct",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1], "y_pred": [0]}),
            AccuracyResult(num_correct_predictions=0, num_predictions=1),
            id="single-incorrect",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat"], "y_pred": ["cat"]}),
            AccuracyResult(num_correct_predictions=1, num_predictions=1),
            id="single-str-correct",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat"], "y_pred": ["dog"]}),
            AccuracyResult(num_correct_predictions=0, num_predictions=1),
            id="single-str-incorrect",
        ),
    ],
)
def test_accuracy_frame_single_row(frame: pl.DataFrame, expected: AccuracyResult) -> None:
    assert accuracy(frame, y_true_col="y_true", y_pred_col="y_pred").equal(expected)
