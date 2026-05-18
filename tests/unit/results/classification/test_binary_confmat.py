from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from mlev.results import BinaryConfusionMatrixResult

#################################################
#     Tests for BinaryConfusionMatrixResult     #
#################################################


# ----------------------------------------------------
# Instantiation
# ----------------------------------------------------


def test_binary_confusion_matrix_result_instantiation() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert m.true_positives == 3
    assert m.true_negatives == 4
    assert m.false_positives == 1
    assert m.false_negatives == 2


def test_binary_confusion_matrix_result_frozen() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'true_positives'"):
        m.true_positives = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn"),
    [
        pytest.param(0, 0, 0, 0, id="all-zero"),
        pytest.param(5, 0, 0, 0, id="only-tp"),
        pytest.param(0, 5, 0, 0, id="only-tn"),
        pytest.param(0, 0, 5, 0, id="only-fp"),
        pytest.param(0, 0, 0, 5, id="only-fn"),
        pytest.param(10, 10, 10, 10, id="all-equal"),
    ],
)
def test_binary_confusion_matrix_result_valid(tp: int, tn: int, fp: int, fn: int) -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.true_positives == tp
    assert m.true_negatives == tn
    assert m.false_positives == fp
    assert m.false_negatives == fn


# ----------------------------------------------------
# Validation
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "match"),
    [
        pytest.param(-1, 4, 1, 2, "true_positives", id="negative-tp"),
        pytest.param(3, -1, 1, 2, "true_negatives", id="negative-tn"),
        pytest.param(3, 4, -1, 2, "false_positives", id="negative-fp"),
        pytest.param(3, 4, 1, -1, "false_negatives", id="negative-fn"),
    ],
)
def test_binary_confusion_matrix_result_negative_raises(
    tp: int, tn: int, fp: int, fn: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BinaryConfusionMatrixResult(
            true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
        )


# ----------------------------------------------------
# num_correct_predictions property
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 7, id="standard"),
        pytest.param(0, 0, 0, 0, 0, id="all-zero"),
        pytest.param(5, 0, 0, 0, 5, id="only-tp"),
        pytest.param(0, 5, 0, 0, 5, id="only-tn"),
        pytest.param(0, 0, 5, 3, 0, id="no-correct"),
        pytest.param(10, 10, 0, 0, 20, id="all-correct"),
    ],
)
def test_binary_confusion_matrix_result_num_correct_predictions(
    tp: int, tn: int, fp: int, fn: int, expected: int
) -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.num_correct_predictions == expected


# ----------------------------------------------------
# num_predictions property
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 10, id="standard"),
        pytest.param(0, 0, 0, 0, 0, id="all-zero"),
        pytest.param(5, 0, 0, 0, 5, id="only-tp"),
        pytest.param(0, 0, 3, 7, 10, id="only-incorrect"),
        pytest.param(10, 10, 10, 10, 40, id="all-equal"),
    ],
)
def test_binary_confusion_matrix_result_num_predictions(
    tp: int, tn: int, fp: int, fn: int, expected: int
) -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.num_predictions == expected


# ----------------------------------------------------
# accuracy property
# ----------------------------------------------------


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 0.7, id="standard"),
        pytest.param(10, 0, 0, 0, 1.0, id="all-tp"),
        pytest.param(0, 10, 0, 0, 1.0, id="all-tn"),
        pytest.param(0, 0, 5, 5, 0.0, id="all-incorrect"),
        pytest.param(5, 5, 0, 0, 1.0, id="all-correct"),
        pytest.param(1, 1, 1, 1, 0.5, id="equal-counts"),
    ],
)
def test_binary_confusion_matrix_result_accuracy(
    tp: int, tn: int, fp: int, fn: int, expected: float
) -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.accuracy == pytest.approx(expected)


def test_binary_confusion_matrix_result_accuracy_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert math.isnan(m.accuracy)


# ----------------------------------------------------
# combine
# ----------------------------------------------------


def test_binary_confusion_matrix_result_combine() -> None:
    m1 = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    m2 = BinaryConfusionMatrixResult(
        true_positives=1, true_negatives=2, false_positives=3, false_negatives=4
    )
    assert m1.combine(m2).equal(
        BinaryConfusionMatrixResult(
            true_positives=4, true_negatives=6, false_positives=4, false_negatives=6
        )
    )


def test_binary_confusion_matrix_result_combine_with_zero() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    zero = BinaryConfusionMatrixResult(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert m.combine(zero).equal(m)


def test_binary_confusion_matrix_result_combine_wrong_type_raises() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    with pytest.raises(TypeError, match="Cannot combine"):
        m.combine("not a result")  # type: ignore[arg-type]


# ----------------------------------------------------
# equal
# ----------------------------------------------------


def test_binary_confusion_matrix_result_equal_true() -> None:
    assert BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).equal(
        BinaryConfusionMatrixResult(
            true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
        )
    )


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn"),
    [
        pytest.param(99, 4, 1, 2, id="different-tp"),
        pytest.param(3, 99, 1, 2, id="different-tn"),
        pytest.param(3, 4, 99, 2, id="different-fp"),
        pytest.param(3, 4, 1, 99, id="different-fn"),
    ],
)
def test_binary_confusion_matrix_result_equal_false(tp: int, tn: int, fp: int, fn: int) -> None:
    assert not BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).equal(
        BinaryConfusionMatrixResult(
            true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
        )
    )


def test_binary_confusion_matrix_result_equal_wrong_type() -> None:
    assert not BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).equal("not a result")


# ----------------------------------------------------
# allclose
# ----------------------------------------------------


def test_binary_confusion_matrix_result_allclose_true() -> None:
    assert BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).allclose(
        BinaryConfusionMatrixResult(
            true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
        )
    )


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn"),
    [
        pytest.param(99, 4, 1, 2, id="different-tp"),
        pytest.param(3, 99, 1, 2, id="different-tn"),
        pytest.param(3, 4, 99, 2, id="different-fp"),
        pytest.param(3, 4, 1, 99, id="different-fn"),
    ],
)
def test_binary_confusion_matrix_result_allclose_false(tp: int, tn: int, fp: int, fn: int) -> None:
    assert not BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).allclose(
        BinaryConfusionMatrixResult(
            true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
        )
    )


def test_binary_confusion_matrix_result_allclose_wrong_type() -> None:
    assert not BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).allclose("not a result")


# ----------------------------------------------------
# to_dict
# ----------------------------------------------------


def test_binary_confusion_matrix_result_to_dict() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert m.to_dict() == {
        "accuracy": 0.7,
        "num_correct_predictions": 7,
        "num_predictions": 10,
        "true_positives": 3,
        "true_negatives": 4,
        "false_positives": 1,
        "false_negatives": 2,
    }


def test_binary_confusion_matrix_result_to_dict_prefix() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert m.to_dict(prefix="train_") == {
        "train_accuracy": 0.7,
        "train_num_correct_predictions": 7,
        "train_num_predictions": 10,
        "train_true_positives": 3,
        "train_true_negatives": 4,
        "train_false_positives": 1,
        "train_false_negatives": 2,
    }


def test_binary_confusion_matrix_result_to_dict_suffix() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert m.to_dict(suffix="_val") == {
        "accuracy_val": 0.7,
        "num_correct_predictions_val": 7,
        "num_predictions_val": 10,
        "true_positives_val": 3,
        "true_negatives_val": 4,
        "false_positives_val": 1,
        "false_negatives_val": 2,
    }


def test_binary_confusion_matrix_result_to_dict_prefix_and_suffix() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert m.to_dict(prefix="train_", suffix="_val") == {
        "train_accuracy_val": 0.7,
        "train_num_correct_predictions_val": 7,
        "train_num_predictions_val": 10,
        "train_true_positives_val": 3,
        "train_true_negatives_val": 4,
        "train_false_positives_val": 1,
        "train_false_negatives_val": 2,
    }


def test_binary_confusion_matrix_result_to_dict_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    result = m.to_dict()
    assert math.isnan(result["accuracy"])
    assert result["num_correct_predictions"] == 0
    assert result["num_predictions"] == 0
