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
# Correct predictions
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="numpy-int",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="numpy-float",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat"]),
            np.array(["cat", "dog", "cat"]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="numpy-str",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1, 1]),
            pl.Series("y_pred", [1, 0, 0, 1, 1]),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="polars-int",
        ),
        pytest.param(
            pl.Series("y_true", [1.0, 0.0, 0.0, 1.0, 1.0]),
            pl.Series("y_pred", [1.0, 0.0, 0.0, 1.0, 1.0]),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="polars-float",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "cat"]),
            pl.Series("y_pred", ["cat", "dog", "cat"]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="polars-str",
        ),
        pytest.param(
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="list-int",
        ),
        pytest.param(
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="list-float",
        ),
        pytest.param(
            ["cat", "dog", "cat"],
            ["cat", "dog", "cat"],
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="list-str",
        ),
        pytest.param(
            (1, 0, 0, 1, 1),
            (1, 0, 0, 1, 1),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="tuple-int",
        ),
        pytest.param(
            (1.0, 0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0, 1.0),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="tuple-float",
        ),
        pytest.param(
            ("cat", "dog", "cat"),
            ("cat", "dog", "cat"),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="tuple-str",
        ),
    ],
)
def test_accuracy_all_correct(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([0, 1, 1, 0, 0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="numpy-int",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="numpy-float",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat"]),
            np.array(["dog", "cat", "dog"]),
            AccuracyResult(num_correct_predictions=0, num_predictions=3),
            id="numpy-str",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1, 1]),
            pl.Series("y_pred", [0, 1, 1, 0, 0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="polars-int",
        ),
        pytest.param(
            pl.Series("y_true", [1.0, 0.0, 0.0, 1.0, 1.0]),
            pl.Series("y_pred", [0.0, 1.0, 1.0, 0.0, 0.0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="polars-float",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "cat"]),
            pl.Series("y_pred", ["dog", "cat", "dog"]),
            AccuracyResult(num_correct_predictions=0, num_predictions=3),
            id="polars-str",
        ),
        pytest.param(
            [1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0],
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="list-int",
        ),
        pytest.param(
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="list-float",
        ),
        pytest.param(
            ["cat", "dog", "cat"],
            ["dog", "cat", "dog"],
            AccuracyResult(num_correct_predictions=0, num_predictions=3),
            id="list-str",
        ),
        pytest.param(
            (1, 0, 0, 1, 1),
            (0, 1, 1, 0, 0),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="tuple-int",
        ),
        pytest.param(
            (1.0, 0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 0.0, 0.0),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="tuple-float",
        ),
        pytest.param(
            ("cat", "dog", "cat"),
            ("dog", "cat", "dog"),
            AccuracyResult(num_correct_predictions=0, num_predictions=3),
            id="tuple-str",
        ),
    ],
)
def test_accuracy_all_incorrect(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1, 1, 0, 1]),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="numpy-int",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0, 1.0]),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="numpy-float",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["cat", "cat", "cat", "dog"]),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="numpy-str",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [1, 1, 0, 1]),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="polars-int",
        ),
        pytest.param(
            pl.Series("y_true", [1.0, 0.0, 0.0, 1.0]),
            pl.Series("y_pred", [1.0, 1.0, 0.0, 1.0]),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="polars-float",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "cat", "dog"]),
            pl.Series("y_pred", ["cat", "cat", "cat", "dog"]),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="polars-str",
        ),
        pytest.param(
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="list-int",
        ),
        pytest.param(
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="list-float",
        ),
        pytest.param(
            ["cat", "dog", "cat", "dog"],
            ["cat", "cat", "cat", "dog"],
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="list-str",
        ),
        pytest.param(
            (1, 0, 0, 1),
            (1, 1, 0, 1),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="tuple-int",
        ),
        pytest.param(
            (1.0, 0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="tuple-float",
        ),
        pytest.param(
            ("cat", "dog", "cat", "dog"),
            ("cat", "cat", "cat", "dog"),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="tuple-str",
        ),
    ],
)
def test_accuracy_partial_correct(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(expected)


# ----------------------------------------------------
# missing_policy='propagate' (default)
# ----------------------------------------------------


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
            np.array([1.0, np.nan, 0.0, 1.0]),
            np.array([1.0, np.nan, np.nan, 1.0]),
            id="numpy-nan-in-both",
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
        pytest.param(
            pl.Series("y_true", [1, None, 0, 1]),
            pl.Series("y_pred", [1, None, None, 1]),
            id="polars-null-in-both",
        ),
    ],
)
def test_accuracy_missing_propagate(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4),
        equal_nan=True,
    )


# ----------------------------------------------------
# missing_policy='omit'
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1.0, 0.0, np.nan, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="numpy-nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, np.nan, 1.0]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="numpy-nan-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, 0.0, np.nan, 1.0, 1.0, np.nan]),
            np.array([1.0, np.nan, 0.0, 1.0, 0.0, np.nan]),
            AccuracyResult(num_correct_predictions=2, num_predictions=3),
            id="numpy-nan-in-both",
        ),
        pytest.param(
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
            id="numpy-all-nan",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, None, 1]),
            pl.Series("y_pred", [1, 0, 0, 1]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="polars-null-in-y_true",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [1, 0, None, 1]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="polars-null-in-y_pred",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, None, 1, 1, None]),
            pl.Series("y_pred", [1, None, 0, 1, 0, None]),
            AccuracyResult(num_correct_predictions=2, num_predictions=3),
            id="polars-null-in-both",
        ),
        pytest.param(
            pl.Series("y_true", [None, None], dtype=pl.Int64),
            pl.Series("y_pred", [None, None], dtype=pl.Int64),
            AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
            id="polars-all-null",
        ),
    ],
)
def test_accuracy_missing_omit(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred, missing_policy="omit").equal(
        expected, equal_nan=True
    )


# ----------------------------------------------------
# missing_policy='raise'
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1, 0, 0, 1]),
            AccuracyResult(num_correct_predictions=4, num_predictions=4),
            id="numpy",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0, 1]),
            pl.Series("y_pred", [1, 0, 0, 1]),
            AccuracyResult(num_correct_predictions=4, num_predictions=4),
            id="polars",
        ),
    ],
)
def test_accuracy_no_missing_raise(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred, missing_policy="raise").equal(expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        pytest.param(
            np.array([1.0, np.nan, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            "'y_true'",
            id="numpy-nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, np.nan, 0.0]),
            "'y_pred'",
            id="numpy-nan-in-y_pred",
        ),
        pytest.param(
            pl.Series("y_true", [1, None, 0]),
            pl.Series("y_pred", [1, 0, 0]),
            "y_true",
            id="polars-null-in-y_true",
        ),
        pytest.param(
            pl.Series("y_true", [1, 0, 0]),
            pl.Series("y_pred", [1, None, 0]),
            "y_pred",
            id="polars-null-in-y_pred",
        ),
    ],
)
def test_accuracy_missing_raise(y_true: ArrayLike, y_pred: ArrayLike, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        accuracy(y_true=y_true, y_pred=y_pred, missing_policy="raise")


# ----------------------------------------------------
# Edge cases
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(np.array([]), np.array([]), id="numpy"),
        pytest.param(
            pl.Series("y_true", [], dtype=pl.Int64),
            pl.Series("y_pred", [], dtype=pl.Int64),
            id="polars",
        ),
        pytest.param([], [], id="list"),
        pytest.param((), (), id="tuple"),
    ],
)
def test_accuracy_empty(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1]),
            np.array([1]),
            AccuracyResult(num_correct_predictions=1, num_predictions=1),
            id="numpy-correct",
        ),
        pytest.param(
            np.array([1]),
            np.array([0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=1),
            id="numpy-incorrect",
        ),
        pytest.param(
            pl.Series("y_true", [1]),
            pl.Series("y_pred", [1]),
            AccuracyResult(num_correct_predictions=1, num_predictions=1),
            id="polars-correct",
        ),
        pytest.param(
            pl.Series("y_true", [1]),
            pl.Series("y_pred", [0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=1),
            id="polars-incorrect",
        ),
    ],
)
def test_accuracy_single_element(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([0.0, -0.0, 1.0]),
            np.array([-0.0, 0.0, 1.0]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="numpy-negative-zero",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="numpy-float-exact",
        ),
        pytest.param(
            pl.Series("y_true", [0.0, -0.0, 1.0]),
            pl.Series("y_pred", [-0.0, 0.0, 1.0]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="polars-negative-zero",
        ),
    ],
)
def test_accuracy_float_edge_cases(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(expected)


# ----------------------------------------------------
# Multiclass string labels
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array(["cat", "dog", "bird", "fish", "cat"]),
            np.array(["cat", "dog", "bird", "fish", "cat"]),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="numpy-all-correct",
        ),
        pytest.param(
            np.array(["cat", "dog", "bird", "fish", "cat"]),
            np.array(["dog", "cat", "fish", "bird", "dog"]),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="numpy-all-incorrect",
        ),
        pytest.param(
            np.array(["cat", "dog", "bird", "fish"]),
            np.array(["cat", "cat", "bird", "bird"]),
            AccuracyResult(num_correct_predictions=2, num_predictions=4),
            id="numpy-partial",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "bird", "fish", "cat"]),
            pl.Series("y_pred", ["cat", "dog", "bird", "fish", "cat"]),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="polars-all-correct",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "bird", "fish", "cat"]),
            pl.Series("y_pred", ["dog", "cat", "fish", "bird", "dog"]),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="polars-all-incorrect",
        ),
        pytest.param(
            pl.Series("y_true", ["cat", "dog", "bird", "fish"]),
            pl.Series("y_pred", ["cat", "cat", "bird", "bird"]),
            AccuracyResult(num_correct_predictions=2, num_predictions=4),
            id="polars-partial",
        ),
        pytest.param(
            ["cat", "dog", "bird", "fish", "cat"],
            ["cat", "dog", "bird", "fish", "cat"],
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="list-all-correct",
        ),
        pytest.param(
            ["cat", "dog", "bird", "fish"],
            ["cat", "cat", "bird", "bird"],
            AccuracyResult(num_correct_predictions=2, num_predictions=4),
            id="list-partial",
        ),
        pytest.param(
            ("cat", "dog", "bird", "fish", "cat"),
            ("cat", "dog", "bird", "fish", "cat"),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="tuple-all-correct",
        ),
        pytest.param(
            ("cat", "dog", "bird", "fish"),
            ("cat", "cat", "bird", "bird"),
            AccuracyResult(num_correct_predictions=2, num_predictions=4),
            id="tuple-partial",
        ),
    ],
)
def test_accuracy_multiclass_string(
    y_true: ArrayLike, y_pred: ArrayLike, expected: AccuracyResult
) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(expected)
