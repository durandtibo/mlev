from __future__ import annotations

import polars as pl
import pytest

from mlev.functional.frame import binary_confusion_matrix
from mlev.results import BinaryConfusionMatrixResult

############################################
#    Tests for binary_confusion_matrix     #
############################################


# --- basic correctness ---


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 1, 1, 0, 0, 1, 0], "y_pred": [1, 0, 1, 0, 1, 0, 1, 0]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=3, true_negatives=3, false_positives=1, false_negatives=1
            ),
            id="int-partial",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 1, 0]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=2, false_positives=0, false_negatives=0
            ),
            id="int-all-correct",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 1, 0], "y_pred": [0, 1, 0, 1]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=0, true_negatives=0, false_positives=2, false_negatives=2
            ),
            id="int-all-incorrect",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1.0, 0.0, 1.0, 0.0], "y_pred": [1.0, 0.0, 0.0, 1.0]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
            ),
            id="float-partial",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": ["cat", "dog", "cat", "dog"], "y_pred": ["cat", "dog", "dog", "dog"]}
            ),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=1, false_negatives=0
            ),
            id="str-partial",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": ["cat", "dog", "cat", "dog"], "y_pred": ["cat", "dog", "cat", "dog"]}
            ),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=2, false_positives=0, false_negatives=0
            ),
            id="str-all-correct",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": ["cat", "dog", "cat", "dog"], "y_pred": ["dog", "cat", "dog", "cat"]}
            ),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=0, true_negatives=0, false_positives=2, false_negatives=2
            ),
            id="str-all-incorrect",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": [True, False, True, False], "y_pred": [True, False, False, True]}
            ),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
            ),
            id="bool-partial",
        ),
    ],
)
def test_binary_confusion_matrix_frame(
    frame: pl.DataFrame, expected: BinaryConfusionMatrixResult
) -> None:
    assert binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred").equal(expected)


# --- single class ---


def test_binary_confusion_matrix_frame_single_class() -> None:
    frame = pl.DataFrame({"y_true": [1, 1, 1], "y_pred": [1, 1, 1]})
    assert binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=3, true_negatives=0, false_positives=0, false_negatives=0
        )
    )


# --- extra columns are ignored ---


def test_binary_confusion_matrix_frame_extra_columns_ignored() -> None:
    frame = pl.DataFrame(
        {
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 0, 1],
            "extra": [10, 20, 30, 40],
        }
    )
    assert binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
        )
    )


def test_binary_confusion_matrix_frame_extra_column_with_missing_ignored() -> None:
    frame = pl.DataFrame(
        {
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 0, 1],
            "extra": [10, None, 30, 40],
        }
    )
    assert binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
        )
    )


# --- missing_policy='propagate' (default) ---


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, None, 1], "y_pred": [1, 0, 0, 1]}),
            id="null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1], "y_pred": [1, 0, None, 1]}),
            id="null-in-y_pred",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, None, 0, 1], "y_pred": [1, None, None, 1]}),
            id="null-in-both",
        ),
    ],
)
def test_binary_confusion_matrix_frame_missing_propagate(frame: pl.DataFrame) -> None:
    import math

    result = binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred")
    assert math.isnan(result.true_positives)
    assert math.isnan(result.accuracy)


# --- missing_policy='omit' ---


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, None, 1], "y_pred": [1, 0, 0, 1]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=0, false_negatives=0
            ),
            id="null-in-y_true",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, 0, 1], "y_pred": [1, 0, None, 1]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=0, false_negatives=0
            ),
            id="null-in-y_pred",
        ),
        pytest.param(
            pl.DataFrame({"y_true": [1, 0, None, 1, 1, None], "y_pred": [1, None, 0, 1, 0, None]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=0, false_positives=0, false_negatives=1
            ),
            id="null-in-both",
        ),
        pytest.param(
            pl.DataFrame(
                {"y_true": [None, None], "y_pred": [None, None]},
                schema={"y_true": pl.Int64, "y_pred": pl.Int64},
            ),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
            ),
            id="all-null",
        ),
        pytest.param(
            pl.DataFrame({"y_true": ["cat", None, "cat"], "y_pred": ["cat", "dog", "cat"]}),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=0, false_positives=0, false_negatives=0
            ),
            id="str-null-in-y_true",
        ),
    ],
)
def test_binary_confusion_matrix_frame_missing_omit(
    frame: pl.DataFrame, expected: BinaryConfusionMatrixResult
) -> None:
    assert binary_confusion_matrix(
        frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="omit"
    ).equal(expected, equal_nan=True)


# --- missing_policy='raise' ---


def test_binary_confusion_matrix_frame_no_missing_raise() -> None:
    frame = pl.DataFrame({"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 0, 1]})
    assert binary_confusion_matrix(
        frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="raise"
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
        )
    )


def test_binary_confusion_matrix_frame_missing_in_y_true_raise() -> None:
    frame = pl.DataFrame({"y_true": [1, None, 0], "y_pred": [1, 0, 0]})
    with pytest.raises(ValueError, match="input contains at least one missing value"):
        binary_confusion_matrix(
            frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="raise"
        )


def test_binary_confusion_matrix_frame_missing_in_y_pred_raise() -> None:
    frame = pl.DataFrame({"y_true": [1, 0, 0], "y_pred": [1, None, 0]})
    with pytest.raises(ValueError, match="input contains at least one missing value"):
        binary_confusion_matrix(
            frame, y_true_col="y_true", y_pred_col="y_pred", missing_policy="raise"
        )


# --- betas ---


def test_binary_confusion_matrix_frame_multiple_betas() -> None:
    frame = pl.DataFrame({"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 0, 1]})
    result = binary_confusion_matrix(
        frame, y_true_col="y_true", y_pred_col="y_pred", betas=[0.5, 1.0, 2.0]
    )
    assert set(result.f_beta_scores.keys()) == {0.5, 1.0, 2.0}


# --- edge cases ---


def test_binary_confusion_matrix_frame_empty() -> None:
    frame = pl.DataFrame(
        {"y_true": [], "y_pred": []}, schema={"y_true": pl.Int64, "y_pred": pl.Int64}
    )
    assert binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
        ),
        equal_nan=True,
    )


def test_binary_confusion_matrix_frame_single_row_correct() -> None:
    frame = pl.DataFrame({"y_true": [1], "y_pred": [1]})
    assert binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=0, false_positives=0, false_negatives=0
        )
    )


def test_binary_confusion_matrix_frame_single_row_incorrect() -> None:
    frame = pl.DataFrame({"y_true": [1, 0], "y_pred": [0, 0]})
    assert binary_confusion_matrix(frame, y_true_col="y_true", y_pred_col="y_pred").equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=1, false_positives=0, false_negatives=1
        )
    )
