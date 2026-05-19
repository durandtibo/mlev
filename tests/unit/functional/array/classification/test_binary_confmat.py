import numpy as np
import polars as pl
import pytest

from mlev.functional.array import binary_confusion_matrix
from mlev.functional.array.classification.binary_confmat import compute_confusion_matrix
from mlev.results import BinaryConfusionMatrixResult
from mlev.typing import ArrayLike

MISSING_POLICIES = ["omit", "propagate", "raise"]

##############################################
#     Tests for compute_confusion_matrix     #
##############################################


# --- has_missing=True ---


def test_compute_confusion_matrix_has_missing() -> None:
    result = compute_confusion_matrix(
        y_true=np.array([1, 0, 1]),
        y_pred=np.array([1, 0, 0]),
        has_missing=True,
        betas=[1.0],
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=float("nan"),
            true_negatives=float("nan"),
            false_positives=float("nan"),
            false_negatives=float("nan"),
        ),
        equal_nan=True,
    )


def test_compute_confusion_matrix_has_missing_all_nan_metrics() -> None:
    import math

    result = compute_confusion_matrix(
        y_true=np.array([1, 0, 1]),
        y_pred=np.array([1, 0, 0]),
        has_missing=True,
        betas=[1.0],
    )
    assert math.isnan(result.accuracy)
    assert math.isnan(result.precision)
    assert math.isnan(result.recall)
    assert math.isnan(result.specificity)


# --- empty arrays ---


def test_compute_confusion_matrix_empty_array() -> None:
    result = compute_confusion_matrix(
        y_true=np.array([]),
        y_pred=np.array([]),
        has_missing=False,
        betas=[1.0],
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
        ),
        equal_nan=True,
    )


def test_compute_confusion_matrix_empty_polars_series() -> None:
    result = compute_confusion_matrix(
        y_true=pl.Series("y_true", [], dtype=pl.Int64),
        y_pred=pl.Series("y_pred", [], dtype=pl.Int64),
        has_missing=False,
        betas=[1.0],
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
        ),
        equal_nan=True,
    )


# --- single class (assumed positive) ---


def test_compute_confusion_matrix_single_class_all_correct() -> None:
    # All positive, all predicted positive
    result = compute_confusion_matrix(
        y_true=np.array([1, 1, 1]),
        y_pred=np.array([1, 1, 1]),
        has_missing=False,
        betas=[1.0],
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=3, true_negatives=0, false_positives=0, false_negatives=0
        )
    )


def test_compute_confusion_matrix_single_class_string_labels() -> None:
    result = compute_confusion_matrix(
        y_true=np.array(["cat", "cat", "cat"]),
        y_pred=np.array(["cat", "cat", "cat"]),
        has_missing=False,
        betas=[1.0],
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=3, true_negatives=0, false_positives=0, false_negatives=0
        )
    )


# --- standard binary cases (numpy) ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=3, true_negatives=3, false_positives=1, false_negatives=1
            ),
            id="standard-int",
        ),
        pytest.param(
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
            ),
            id="standard-float",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["cat", "dog", "dog", "dog"]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=1, false_negatives=0
            ),
            id="standard-str",
        ),
        pytest.param(
            np.array([True, False, True, False]),
            np.array([True, False, False, True]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
            ),
            id="standard-bool",
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1, 0, 0, 1]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=2, false_positives=0, false_negatives=0
            ),
            id="all-correct",
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([0, 1, 1, 0]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=0, true_negatives=0, false_positives=2, false_negatives=2
            ),
            id="all-incorrect",
        ),
    ],
)
def test_compute_confusion_matrix_array(
    y_true: np.ndarray, y_pred: np.ndarray, expected: BinaryConfusionMatrixResult
) -> None:
    result = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, has_missing=False, betas=[1.0])
    assert result.equal(expected)


# --- standard binary cases (polars) ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            pl.Series("y_true", [1, 0, 1, 1, 0, 0, 1, 0]),
            pl.Series("y_pred", [1, 0, 1, 0, 1, 0, 1, 0]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=3, true_negatives=3, false_positives=1, false_negatives=1
            ),
            id="standard-int",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "cat", "dog"]),
            pl.Series("y_pred", ["cat", "dog", "dog", "dog"]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=1, false_negatives=0
            ),
            id="standard-str",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [1, 0, 0, 1]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=2, false_positives=0, false_negatives=0
            ),
            id="all-correct",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [0, 1, 1, 0]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=0, true_negatives=0, false_positives=2, false_negatives=2
            ),
            id="all-incorrect",
        ),
    ],
)
def test_compute_confusion_matrix_series(
    y_true: pl.Series, y_pred: pl.Series, expected: BinaryConfusionMatrixResult
) -> None:
    result = compute_confusion_matrix(y_true=y_true, y_pred=y_pred, has_missing=False, betas=[1.0])
    assert result.equal(expected)


# --- betas ---


def test_compute_confusion_matrix_multiple_betas() -> None:
    result = compute_confusion_matrix(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 0, 1]),
        has_missing=False,
        betas=[0.5, 1.0, 2.0],
    )
    assert set(result.f_beta_scores.keys()) == {0.5, 1.0, 2.0}


############################################
#    Tests for binary_confusion_matrix     #
############################################


# --- basic correctness across input types ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(np.array([1, 0, 1, 0]), np.array([1, 0, 0, 1]), id="numpy"),
        pytest.param(
            pl.Series("y_true", [1, 0, 1, 0]), pl.Series("y_pred", [1, 0, 0, 1]), id="polars"
        ),
        pytest.param([1, 0, 1, 0], [1, 0, 0, 1], id="list"),
        pytest.param((1, 0, 1, 0), (1, 0, 0, 1), id="tuple"),
    ],
)
def test_binary_confusion_matrix_input_types(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    result = binary_confusion_matrix(y_true=y_true, y_pred=y_pred)
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
        )
    )


# --- int labels ---


def test_binary_confusion_matrix_int_all_correct() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 1, 0]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=2, true_negatives=2, false_positives=0, false_negatives=0
        )
    )


def test_binary_confusion_matrix_int_all_incorrect() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([0, 1, 0, 1]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=0, false_positives=2, false_negatives=2
        )
    )


def test_binary_confusion_matrix_int_partial() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        y_pred=np.array([1, 0, 1, 0, 1, 0, 1, 0]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=3, true_negatives=3, false_positives=1, false_negatives=1
        )
    )


# --- float labels ---


def test_binary_confusion_matrix_float() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1.0, 0.0, 1.0, 0.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
        )
    )


# --- string labels ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            ["cat", "dog", "cat", "dog"],
            ["cat", "dog", "cat", "dog"],
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=2, false_positives=0, false_negatives=0
            ),
            id="str-all-correct",
        ),
        pytest.param(
            ["cat", "dog", "cat", "dog"],
            ["dog", "cat", "dog", "cat"],
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=0, true_negatives=0, false_positives=2, false_negatives=2
            ),
            id="str-all-incorrect",
        ),
        pytest.param(
            ["cat", "dog", "cat", "dog"],
            ["cat", "dog", "dog", "dog"],
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=1, false_negatives=0
            ),
            id="str-partial",
        ),
    ],
)
def test_binary_confusion_matrix_string(
    y_true: ArrayLike, y_pred: ArrayLike, expected: BinaryConfusionMatrixResult
) -> None:
    assert binary_confusion_matrix(y_true=y_true, y_pred=y_pred).equal(expected)


# --- bool labels ---


def test_binary_confusion_matrix_bool() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([True, False, True, False]),
        y_pred=np.array([True, False, False, True]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
        )
    )


# --- single class ---


def test_binary_confusion_matrix_single_class_array() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1, 1, 1]),
        y_pred=np.array([1, 1, 1]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=3, true_negatives=0, false_positives=0, false_negatives=0
        )
    )


def test_binary_confusion_matrix_single_class_series() -> None:
    assert binary_confusion_matrix(
        y_true=pl.Series("y_true", [1, 1, 1]),
        y_pred=pl.Series("y_pred", [1, 1, 1]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=3, true_negatives=0, false_positives=0, false_negatives=0
        )
    )


# --- missing_policy='propagate' (default) ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([1.0, 0.0, np.nan, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            id="numpy-nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, np.nan, 1.0]),
            id="numpy-nan-in-y_pred",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, None, 1]),
            pl.Series("y_pred", [1, 0, 0, 1]),
            id="polars-null-in-y_true",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [1, 0, None, 1]),
            id="polars-null-in-y_pred",
        ),
    ],
)
def test_binary_confusion_matrix_missing_propagate(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    import math

    result = binary_confusion_matrix(y_true=y_true, y_pred=y_pred)
    assert math.isnan(result.true_positives)
    assert math.isnan(result.accuracy)


# --- missing_policy='omit' ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1.0, 0.0, np.nan, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=0, false_negatives=0
            ),
            id="numpy-nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, np.nan, 1.0]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=0, false_negatives=0
            ),
            id="numpy-nan-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, 0.0, np.nan, 1.0, 1.0, np.nan]),
            np.array([1.0, np.nan, 0.0, 1.0, 0.0, np.nan]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=0, false_positives=0, false_negatives=1
            ),
            id="numpy-nan-in-both",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, None, 1]),
            pl.Series("y_pred", [1, 0, 0, 1]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=0, false_negatives=0
            ),
            id="polars-null-in-y_true",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [1, 0, None, 1]),
            BinaryConfusionMatrixResult.from_confusion_matrix(
                true_positives=2, true_negatives=1, false_positives=0, false_negatives=0
            ),
            id="polars-null-in-y_pred",
        ),
    ],
)
def test_binary_confusion_matrix_missing_omit(
    y_true: ArrayLike, y_pred: ArrayLike, expected: BinaryConfusionMatrixResult
) -> None:
    assert binary_confusion_matrix(y_true=y_true, y_pred=y_pred, missing_policy="omit").equal(
        expected
    )


def test_binary_confusion_matrix_all_missing_omit() -> None:
    result = binary_confusion_matrix(
        y_true=np.array([np.nan, np.nan]),
        y_pred=np.array([np.nan, np.nan]),
        missing_policy="omit",
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
        ),
        equal_nan=True,
    )


# --- missing_policy='raise' ---


def test_binary_confusion_matrix_no_missing_raise() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 0, 1]),
        missing_policy="raise",
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=1, false_positives=1, false_negatives=1
        )
    )


def test_binary_confusion_matrix_missing_in_y_true_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one missing value"):
        binary_confusion_matrix(
            y_true=np.array([1.0, np.nan, 0.0]),
            y_pred=np.array([1.0, 0.0, 0.0]),
            missing_policy="raise",
        )


def test_binary_confusion_matrix_missing_in_y_pred_raise() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one missing value"):
        binary_confusion_matrix(
            y_true=np.array([1.0, 0.0, 0.0]),
            y_pred=np.array([1.0, np.nan, 0.0]),
            missing_policy="raise",
        )


def test_binary_confusion_matrix_polars_missing_raise() -> None:
    with pytest.raises(ValueError, match="y_true contains at least one missing value"):
        binary_confusion_matrix(
            y_true=pl.Series("y_true", [1, None, 0]),
            y_pred=pl.Series("y_pred", [1, 0, 0]),
            missing_policy="raise",
        )


# --- betas ---


def test_binary_confusion_matrix_multiple_betas() -> None:
    result = binary_confusion_matrix(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 0, 1]),
        betas=[0.5, 1.0, 2.0],
    )
    assert set(result.f_beta_scores.keys()) == {0.5, 1.0, 2.0}


# --- edge cases ---


def test_binary_confusion_matrix_empty_array() -> None:
    result = binary_confusion_matrix(
        y_true=np.array([], dtype=float),
        y_pred=np.array([], dtype=float),
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
        ),
        equal_nan=True,
    )


def test_binary_confusion_matrix_empty_polars_series() -> None:
    result = binary_confusion_matrix(
        y_true=pl.Series("y_true", [], dtype=pl.Int64),
        y_pred=pl.Series("y_pred", [], dtype=pl.Int64),
    )
    assert result.equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
        ),
        equal_nan=True,
    )


def test_binary_confusion_matrix_single_element_correct() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1]),
        y_pred=np.array([1]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=1, true_negatives=0, false_positives=0, false_negatives=0
        )
    )


def test_binary_confusion_matrix_single_element_incorrect() -> None:
    assert binary_confusion_matrix(
        y_true=np.array([1, 0]),
        y_pred=np.array([0, 0]),
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=0, true_negatives=1, false_positives=0, false_negatives=1
        )
    )
