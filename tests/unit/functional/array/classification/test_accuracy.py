import numpy as np
import polars as pl
import pytest

from mlev.functional.array import accuracy
from mlev.results import AccuracyResult
from mlev.typing import ArrayLike

##############################
#     Tests for accuracy     #
##############################


# ----------------------------------------------------
# numpy arrays
# ----------------------------------------------------


# --- correct predictions ---


def test_accuracy_array_all_correct() -> None:
    assert accuracy(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
    ).equal(AccuracyResult(num_correct_predictions=5, num_predictions=5))


def test_accuracy_array_all_incorrect() -> None:
    assert accuracy(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 0, 0]),
    ).equal(AccuracyResult(num_correct_predictions=0, num_predictions=5))


def test_accuracy_array_partial_correct() -> None:
    assert accuracy(
        y_true=np.array([1, 0, 0, 1]),
        y_pred=np.array([1, 1, 0, 1]),
    ).equal(AccuracyResult(num_correct_predictions=3, num_predictions=4))


# --- missing_policy='propagate' (default) ---


def test_accuracy_array_missing_in_y_true_propagate() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, np.nan, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4), equal_nan=True)


def test_accuracy_array_missing_in_y_pred_propagate() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, np.nan, 1.0]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4), equal_nan=True)


def test_accuracy_array_missing_in_both_propagate() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, np.nan, 1.0]),
        y_pred=np.array([1.0, np.nan, 0.0, 1.0]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4), equal_nan=True)


# --- missing_policy='omit' ---


def test_accuracy_array_missing_in_y_true_omit() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, np.nan, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=3, num_predictions=3))


def test_accuracy_array_missing_in_y_pred_omit() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, np.nan, 1.0]),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=3, num_predictions=3))


def test_accuracy_array_missing_in_both_omit() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, np.nan, 1.0, 1.0, np.nan]),
        y_pred=np.array([1.0, np.nan, 0.0, 1.0, 0.0, np.nan]),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=2, num_predictions=3))


def test_accuracy_array_all_missing_omit() -> None:
    assert accuracy(
        y_true=np.array([np.nan, np.nan]),
        y_pred=np.array([np.nan, np.nan]),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0), equal_nan=True)


# --- missing_policy='raise' ---


def test_accuracy_array_no_missing_raise() -> None:
    assert accuracy(
        y_true=np.array([1, 0, 0, 1]),
        y_pred=np.array([1, 0, 0, 1]),
        missing_policy="raise",
    ).equal(AccuracyResult(num_correct_predictions=4, num_predictions=4))


def test_accuracy_array_missing_in_y_true_raise() -> None:
    with pytest.raises(ValueError, match="'y_true'"):
        accuracy(
            y_true=np.array([1.0, np.nan, 0.0]),
            y_pred=np.array([1.0, 0.0, 0.0]),
            missing_policy="raise",
        )


def test_accuracy_array_missing_in_y_pred_raise() -> None:
    with pytest.raises(ValueError, match="'y_pred'"):
        accuracy(
            y_true=np.array([1.0, 0.0, 0.0]),
            y_pred=np.array([1.0, np.nan, 0.0]),
            missing_policy="raise",
        )


# --- edge cases ---


def test_accuracy_array_empty() -> None:
    assert accuracy(
        y_true=np.array([]),
        y_pred=np.array([]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0), equal_nan=True)


def test_accuracy_array_single_correct() -> None:
    assert accuracy(
        y_true=np.array([1]),
        y_pred=np.array([1]),
    ).equal(AccuracyResult(num_correct_predictions=1, num_predictions=1))


def test_accuracy_array_single_incorrect() -> None:
    assert accuracy(
        y_true=np.array([1]),
        y_pred=np.array([0]),
    ).equal(AccuracyResult(num_correct_predictions=0, num_predictions=1))


# ----------------------------------------------------
# polars Series
# ----------------------------------------------------

# --- correct predictions ---


def test_accuracy_series_all_correct() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, 0, 1, 1]),
        y_pred=pl.Series("y_pred", [1, 0, 0, 1, 1]),
    ).equal(AccuracyResult(num_correct_predictions=5, num_predictions=5))


def test_accuracy_series_all_incorrect() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, 0, 1, 1]),
        y_pred=pl.Series("y_pred", [0, 1, 1, 0, 0]),
    ).equal(AccuracyResult(num_correct_predictions=0, num_predictions=5))


def test_accuracy_series_partial_correct() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, 0, 1]),
        y_pred=pl.Series("y_pred", [1, 1, 0, 1]),
    ).equal(AccuracyResult(num_correct_predictions=3, num_predictions=4))


# --- missing_policy='propagate' (default) ---


def test_accuracy_series_missing_in_y_true_propagate() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, None, 1]),
        y_pred=pl.Series("y_pred", [1, 0, 0, 1]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4), equal_nan=True)


def test_accuracy_series_missing_in_y_pred_propagate() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, 0, 1]),
        y_pred=pl.Series("y_pred", [1, 0, None, 1]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4), equal_nan=True)


def test_accuracy_series_missing_in_both_propagate() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, None, 1]),
        y_pred=pl.Series("y_pred", [1, None, 0, 1]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4), equal_nan=True)


# --- missing_policy='omit' ---


def test_accuracy_series_missing_in_y_true_omit() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, None, 1]),
        y_pred=pl.Series("y_pred", [1, 0, 0, 1]),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=3, num_predictions=3))


def test_accuracy_series_missing_in_y_pred_omit() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, 0, 1]),
        y_pred=pl.Series("y_pred", [1, 0, None, 1]),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=3, num_predictions=3))


def test_accuracy_series_missing_in_both_omit() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, None, 1, 1, None]),
        y_pred=pl.Series("y_pred", [1, None, 0, 1, 0, None]),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=2, num_predictions=3))


def test_accuracy_series_all_missing_omit() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [None, None], dtype=pl.Int64),
        y_pred=pl.Series("y_pred", [None, None], dtype=pl.Int64),
        missing_policy="omit",
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0), equal_nan=True)


# --- missing_policy='raise' ---


def test_accuracy_series_no_missing_raise() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, 0, 1]),
        y_pred=pl.Series("y_pred", [1, 0, 0, 1]),
        missing_policy="raise",
    ).equal(AccuracyResult(num_correct_predictions=4, num_predictions=4))


def test_accuracy_series_missing_in_y_true_raise() -> None:
    with pytest.raises(ValueError, match="y_true"):
        accuracy(
            y_true=pl.Series("y_true", [1, None, 0]),
            y_pred=pl.Series("y_pred", [1, 0, 0]),
            missing_policy="raise",
        )


def test_accuracy_series_missing_in_y_pred_raise() -> None:
    with pytest.raises(ValueError, match="y_pred"):
        accuracy(
            y_true=pl.Series("y_true", [1, 0, 0]),
            y_pred=pl.Series("y_pred", [1, None, 0]),
            missing_policy="raise",
        )


# --- edge cases ---


def test_accuracy_series_empty() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [], dtype=pl.Int64),
        y_pred=pl.Series("y_pred", [], dtype=pl.Int64),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0), equal_nan=True)


def test_accuracy_series_single_correct() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1]),
        y_pred=pl.Series("y_pred", [1]),
    ).equal(AccuracyResult(num_correct_predictions=1, num_predictions=1))


def test_accuracy_series_single_incorrect() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1]),
        y_pred=pl.Series("y_pred", [0]),
    ).equal(AccuracyResult(num_correct_predictions=0, num_predictions=1))


# ----------------------------------------------------
# Tests for accuracy consistency across input types
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1, 1]),
            pl.Series("y_pred", [1, 0, 0, 1, 1]),
            id="polars",
        ),
        pytest.param(
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            id="list",
        ),
        pytest.param(
            (1, 0, 0, 1, 1),
            (1, 0, 0, 1, 1),
            id="tuple",
        ),
    ],
)
def test_accuracy_consistent_across_input_types_all_correct(
    y_true: ArrayLike, y_pred: ArrayLike
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=5, num_predictions=5)
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([0, 1, 1, 0, 0]),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1, 1]),
            pl.Series("y_pred", [0, 1, 1, 0, 0]),
            id="polars",
        ),
        pytest.param(
            [1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0],
            id="list",
        ),
        pytest.param(
            (1, 0, 0, 1, 1),
            (0, 1, 1, 0, 0),
            id="tuple",
        ),
    ],
)
def test_accuracy_consistent_across_input_types_all_incorrect(
    y_true: ArrayLike, y_pred: ArrayLike
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=0, num_predictions=5)
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1, 1, 0, 1]),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [1, 1, 0, 1]),
            id="polars",
        ),
        pytest.param(
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            id="list",
        ),
        pytest.param(
            (1, 0, 0, 1),
            (1, 1, 0, 1),
            id="tuple",
        ),
    ],
)
def test_accuracy_consistent_across_input_types_partial_correct(
    y_true: ArrayLike, y_pred: ArrayLike
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=3, num_predictions=4)
    )


# ----------------------------------------------------
# Tests for accuracy with string labels
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["cat", "dog", "cat", "dog"]),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "cat", "dog"]),
            pl.Series("y_pred", ["cat", "dog", "cat", "dog"]),
            id="polars",
        ),
        pytest.param(
            ["cat", "dog", "cat", "dog"],
            ["cat", "dog", "cat", "dog"],
            id="list",
        ),
        pytest.param(
            ("cat", "dog", "cat", "dog"),
            ("cat", "dog", "cat", "dog"),
            id="tuple",
        ),
    ],
)
def test_accuracy_string_labels_all_correct(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=4, num_predictions=4)
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["dog", "cat", "dog", "cat"]),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "cat", "dog"]),
            pl.Series("y_pred", ["dog", "cat", "dog", "cat"]),
            id="polars",
        ),
        pytest.param(
            ["cat", "dog", "cat", "dog"],
            ["dog", "cat", "dog", "cat"],
            id="list",
        ),
        pytest.param(
            ("cat", "dog", "cat", "dog"),
            ("dog", "cat", "dog", "cat"),
            id="tuple",
        ),
    ],
)
def test_accuracy_string_labels_all_incorrect(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=0, num_predictions=4)
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["cat", "cat", "cat", "dog"]),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "cat", "dog"]),
            pl.Series("y_pred", ["cat", "cat", "cat", "dog"]),
            id="polars",
        ),
        pytest.param(
            ["cat", "dog", "cat", "dog"],
            ["cat", "cat", "cat", "dog"],
            id="list",
        ),
        pytest.param(
            ("cat", "dog", "cat", "dog"),
            ("cat", "cat", "cat", "dog"),
            id="tuple",
        ),
    ],
)
def test_accuracy_string_labels_partial_correct(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=3, num_predictions=4)
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array(["cat", "dog", "bird", "fish", "cat"]),
            np.array(["cat", "dog", "bird", "fish", "cat"]),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "bird", "fish", "cat"]),
            pl.Series("y_pred", ["cat", "dog", "bird", "fish", "cat"]),
            id="polars",
        ),
        pytest.param(
            ["cat", "dog", "bird", "fish", "cat"],
            ["cat", "dog", "bird", "fish", "cat"],
            id="list",
        ),
        pytest.param(
            ("cat", "dog", "bird", "fish", "cat"),
            ("cat", "dog", "bird", "fish", "cat"),
            id="tuple",
        ),
    ],
)
def test_accuracy_string_labels_multiclass(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=5, num_predictions=5)
    )
