from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_allclose

from mlev.results import BinaryConfusionMatrixResult
from mlev.results.classification.binary_confmat import (
    check_betas,
    compute_accuracy,
    compute_f_beta_score,
    compute_precision,
    compute_recall,
    compute_specificity,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

##################################
#     Tests for check_betas      #
##################################


# --- valid inputs ---


def test_check_betas_single_valid() -> None:
    check_betas([1.0])  # should not raise


def test_check_betas_default_f1() -> None:
    check_betas([1.0])  # should not raise


def test_check_betas_zero() -> None:
    check_betas([0.0])  # should not raise — zero is allowed


def test_check_betas_multiple_valid() -> None:
    check_betas([0.5, 1.0, 2.0])  # should not raise


def test_check_betas_empty() -> None:
    check_betas([])  # should not raise — no betas to validate


@pytest.mark.parametrize(
    "betas",
    [
        pytest.param([0.0], id="zero"),
        pytest.param([0.5], id="half"),
        pytest.param([1.0], id="one"),
        pytest.param([2.0], id="two"),
        pytest.param([0.5, 1.0, 2.0], id="multiple"),
        pytest.param([], id="empty"),
        pytest.param((1.0,), id="tuple"),
        pytest.param((0.5, 1.0, 2.0), id="tuple-multiple"),
    ],
)
def test_check_betas_valid(betas: Sequence[float]) -> None:
    check_betas(betas)  # should not raise


# --- invalid inputs ---


def test_check_betas_single_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"beta values must be >= 0, got -1.0"):
        check_betas([-1.0])


def test_check_betas_negative_in_multiple_raises() -> None:
    with pytest.raises(ValueError, match="beta values must be >= 0"):
        check_betas([0.5, -1.0, 2.0])


def test_check_betas_all_negative_raises() -> None:
    with pytest.raises(ValueError, match="beta values must be >= 0"):
        check_betas([-0.5, -1.0, -2.0])


@pytest.mark.parametrize(
    ("betas", "match"),
    [
        pytest.param([-1.0], "beta values must be >= 0, got -1.0", id="minus-one"),
        pytest.param([-0.5], "beta values must be >= 0, got -0.5", id="minus-half"),
        pytest.param([-2.0], "beta values must be >= 0, got -2.0", id="minus-two"),
        pytest.param([1.0, -1.0], "beta values must be >= 0, got -1.0", id="valid-then-invalid"),
        pytest.param([-1.0, 1.0], "beta values must be >= 0, got -1.0", id="invalid-then-valid"),
    ],
)
def test_check_betas_invalid(betas: Sequence[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_betas(betas)


######################################
#     Tests for compute_accuracy     #
######################################


@pytest.mark.parametrize(
    ("num_correct_predictions", "num_predictions", "expected"),
    [
        pytest.param(7, 10, 0.7, id="standard"),
        pytest.param(10, 10, 1.0, id="all-correct"),
        pytest.param(0, 10, 0.0, id="none-correct"),
        pytest.param(1, 1, 1.0, id="single-correct"),
        pytest.param(0, 1, 0.0, id="single-incorrect"),
    ],
)
def test_compute_accuracy(
    num_correct_predictions: int, num_predictions: int, expected: float
) -> None:
    assert compute_accuracy(num_correct_predictions, num_predictions) == (expected)


def test_compute_accuracy_zero_predictions() -> None:
    assert math.isnan(compute_accuracy(num_correct_predictions=0, num_predictions=0))


########################################
#     Tests for compute_precision      #
########################################


@pytest.mark.parametrize(
    ("true_positives", "false_positives", "num_predictions", "expected"),
    [
        pytest.param(3, 1, 10, 0.75, id="standard"),
        pytest.param(5, 0, 10, 1.0, id="no-false-positives"),
        pytest.param(0, 5, 10, 0.0, id="no-true-positives"),
        pytest.param(1, 1, 10, 0.5, id="equal-tp-fp"),
    ],
)
def test_compute_precision(
    true_positives: int, false_positives: int, num_predictions: int, expected: float
) -> None:
    assert compute_precision(true_positives, false_positives, num_predictions) == (expected)


def test_compute_precision_zero_predictions() -> None:
    assert math.isnan(compute_precision(true_positives=3, false_positives=1, num_predictions=0))


def test_compute_precision_zero_denominator() -> None:
    assert compute_precision(true_positives=0, false_positives=0, num_predictions=10) == 0.0


####################################
#     Tests for compute_recall     #
####################################


@pytest.mark.parametrize(
    ("true_positives", "false_negatives", "num_predictions", "expected"),
    [
        pytest.param(3, 2, 10, 0.6, id="standard"),
        pytest.param(5, 0, 10, 1.0, id="no-false-negatives"),
        pytest.param(0, 5, 10, 0.0, id="no-true-positives"),
        pytest.param(1, 1, 10, 0.5, id="equal-tp-fn"),
    ],
)
def test_compute_recall(
    true_positives: int, false_negatives: int, num_predictions: int, expected: float
) -> None:
    assert compute_recall(true_positives, false_negatives, num_predictions) == (expected)


def test_compute_recall_zero_predictions() -> None:
    assert math.isnan(compute_recall(true_positives=3, false_negatives=2, num_predictions=0))


def test_compute_recall_zero_denominator() -> None:
    assert compute_recall(true_positives=0, false_negatives=0, num_predictions=10) == 0.0


#########################################
#     Tests for compute_specificity     #
#########################################


@pytest.mark.parametrize(
    ("true_negatives", "false_positives", "num_predictions", "expected"),
    [
        pytest.param(4, 1, 10, 0.8, id="standard"),
        pytest.param(5, 0, 10, 1.0, id="no-false-positives"),
        pytest.param(0, 5, 10, 0.0, id="no-true-negatives"),
        pytest.param(1, 1, 10, 0.5, id="equal-tn-fp"),
    ],
)
def test_compute_specificity(
    true_negatives: int, false_positives: int, num_predictions: int, expected: float
) -> None:
    assert compute_specificity(true_negatives, false_positives, num_predictions) == (expected)


def test_compute_specificity_zero_predictions() -> None:
    assert math.isnan(compute_specificity(true_negatives=4, false_positives=1, num_predictions=0))


def test_compute_specificity_zero_denominator() -> None:
    assert compute_specificity(true_negatives=0, false_positives=0, num_predictions=10) == 0.0


##########################################
#     Tests for compute_f_beta_score     #
##########################################


@pytest.mark.parametrize(
    ("precision", "recall", "beta", "expected"),
    [
        pytest.param(0.75, 0.6, 1.0, 0.6666666666666665, id="f1-standard"),
        pytest.param(0.75, 0.6, 0.5, 0.7142857142857143, id="f0.5-precision-weighted"),
        pytest.param(0.75, 0.6, 2.0, 0.625, id="f2-recall-weighted"),
        pytest.param(1.0, 1.0, 1.0, 1.0, id="f1-perfect"),
        pytest.param(0.0, 0.0, 1.0, 0.0, id="f1-zero-precision-recall"),
        pytest.param(1.0, 0.0, 1.0, 0.0, id="f1-zero-recall"),
        pytest.param(0.0, 1.0, 1.0, 0.0, id="f1-zero-precision"),
        pytest.param(0.5, 0.5, 0.0, 0.5, id="f0-equals-precision"),
    ],
)
def test_compute_f_beta_score(
    precision: float, recall: float, beta: float, expected: float
) -> None:
    assert compute_f_beta_score(precision, recall, beta) == (expected)


def test_compute_f_beta_score_nan_precision() -> None:
    assert math.isnan(compute_f_beta_score(precision=float("nan"), recall=0.6, beta=1.0))


def test_compute_f_beta_score_nan_recall() -> None:
    assert math.isnan(compute_f_beta_score(precision=0.75, recall=float("nan"), beta=1.0))


def test_compute_f_beta_score_nan_both() -> None:
    assert math.isnan(compute_f_beta_score(precision=float("nan"), recall=float("nan"), beta=1.0))


def test_compute_f_beta_score_negative_beta_raises() -> None:
    with pytest.raises(ValueError, match="beta must be >= 0"):
        compute_f_beta_score(precision=0.75, recall=0.6, beta=-1.0)


#################################################
#     Tests for BinaryConfusionMatrixResult     #
#################################################


# --- precision ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 0.75, id="standard"),
        pytest.param(5, 4, 0, 1, 1.0, id="no-false-positives"),
        pytest.param(0, 4, 5, 1, 0.0, id="no-true-positives"),
        pytest.param(0, 0, 0, 5, 0.0, id="zero-tp-fp"),
    ],
)
def test_binary_confusion_matrix_result_precision(
    tp: int, tn: int, fp: int, fn: int, expected: float
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.precision == (expected)


def test_binary_confusion_matrix_result_precision_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert math.isnan(m.precision)


# --- recall ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 0.6, id="standard"),
        pytest.param(5, 4, 1, 0, 1.0, id="no-false-negatives"),
        pytest.param(0, 4, 1, 5, 0.0, id="no-true-positives"),
        pytest.param(0, 0, 5, 0, 0.0, id="zero-tp-fn"),
    ],
)
def test_binary_confusion_matrix_result_recall(
    tp: int, tn: int, fp: int, fn: int, expected: float
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.recall == (expected)


def test_binary_confusion_matrix_result_recall_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert math.isnan(m.recall)


# --- specificity ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 0.8, id="standard"),
        pytest.param(3, 5, 0, 2, 1.0, id="no-false-positives"),
        pytest.param(3, 0, 5, 2, 0.0, id="no-true-negatives"),
        pytest.param(5, 0, 0, 0, 0.0, id="zero-tn-fp"),
    ],
)
def test_binary_confusion_matrix_result_specificity(
    tp: int, tn: int, fp: int, fn: int, expected: float
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.specificity == (expected)


def test_binary_confusion_matrix_result_specificity_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert math.isnan(m.specificity)


# --- f_beta_scores ---


def test_binary_confusion_matrix_result_f_beta_scores_default() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert m.f_beta_scores == ({1.0: 0.6666666666666665})


@pytest.mark.parametrize(
    ("beta", "expected"),
    [
        pytest.param(0.5, 0.7142857142857143, id="f0.5"),
        pytest.param(1.0, 0.6666666666666665, id="f1"),
        pytest.param(2.0, 0.625, id="f2"),
    ],
)
def test_binary_confusion_matrix_result_f_beta_scores_values(beta: float, expected: float) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3,
        true_negatives=4,
        false_positives=1,
        false_negatives=2,
        betas=[beta],
    )
    assert m.f_beta_scores[beta] == (expected)


def test_binary_confusion_matrix_result_f_beta_scores_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert math.isnan(m.f_beta_scores[1.0])


# --- combine ---


def test_binary_confusion_matrix_result_combine() -> None:
    m1 = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    m2 = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=1, true_negatives=2, false_positives=3, false_negatives=4
    )
    assert m1.combine(m2).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=4, true_negatives=6, false_positives=4, false_negatives=6
        )
    )


def test_binary_confusion_matrix_result_combine_preserves_betas() -> None:
    m1 = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3,
        true_negatives=4,
        false_positives=1,
        false_negatives=2,
        betas=[0.5, 1.0, 2.0],
    )
    m2 = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=1,
        true_negatives=2,
        false_positives=3,
        false_negatives=4,
        betas=[0.5, 1.0, 2.0],
    )
    assert set(m1.combine(m2).f_beta_scores.keys()) == {0.5, 1.0, 2.0}


def test_binary_confusion_matrix_result_combine_with_zero() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    zero = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert m.combine(zero).equal(m)


def test_binary_confusion_matrix_result_combine_wrong_type_raises() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    with pytest.raises(TypeError, match="Cannot combine"):
        m.combine("not a result")  # type: ignore[arg-type]


# --- equal ---


def test_binary_confusion_matrix_result_equal_true() -> None:
    assert BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
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
    assert not BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).equal(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
        )
    )


def test_binary_confusion_matrix_result_equal_wrong_type() -> None:
    assert not BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).equal("not a result")


# --- allclose ---


def test_binary_confusion_matrix_result_allclose_true() -> None:
    assert BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).allclose(
        BinaryConfusionMatrixResult.from_confusion_matrix(
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
    assert not BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).allclose(
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
        )
    )


def test_binary_confusion_matrix_result_allclose_wrong_type() -> None:
    assert not BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    ).allclose("not a result")


# --- to_dict ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "betas", "expected"),
    [
        pytest.param(
            3,
            4,
            1,
            2,
            [1.0],
            {
                "accuracy": 0.7,
                "precision": 0.75,
                "recall": 0.6,
                "specificity": 0.8,
                "f1": 0.6666666666666665,
                "num_correct_predictions": 7,
                "num_predictions": 10,
                "true_positives": 3,
                "true_negatives": 4,
                "false_positives": 1,
                "false_negatives": 2,
            },
            id="standard-f1",
        ),
        pytest.param(
            3,
            4,
            1,
            2,
            [0.5, 1.0, 2.0],
            {
                "accuracy": 0.7,
                "precision": 0.75,
                "recall": 0.6,
                "specificity": 0.8,
                "f0.5": 0.7142857142857143,
                "f1": 0.6666666666666665,
                "f2": 0.625,
                "num_correct_predictions": 7,
                "num_predictions": 10,
                "true_positives": 3,
                "true_negatives": 4,
                "false_positives": 1,
                "false_negatives": 2,
            },
            id="multiple-betas",
        ),
        pytest.param(
            10,
            0,
            0,
            0,
            [1.0],
            {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "specificity": 0.0,
                "f1": 1.0,
                "num_correct_predictions": 10,
                "num_predictions": 10,
                "true_positives": 10,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            id="all-tp",
        ),
        pytest.param(
            0,
            10,
            0,
            0,
            [1.0],
            {
                "accuracy": 1.0,
                "precision": 0.0,
                "recall": 0.0,
                "specificity": 1.0,
                "f1": 0.0,
                "num_correct_predictions": 10,
                "num_predictions": 10,
                "true_positives": 0,
                "true_negatives": 10,
                "false_positives": 0,
                "false_negatives": 0,
            },
            id="all-tn",
        ),
        pytest.param(
            0,
            0,
            5,
            5,
            [1.0],
            {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "specificity": 0.0,
                "f1": 0.0,
                "num_correct_predictions": 0,
                "num_predictions": 10,
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 5,
                "false_negatives": 5,
            },
            id="all-incorrect",
        ),
        pytest.param(
            5,
            5,
            0,
            0,
            [1.0],
            {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "specificity": 1.0,
                "f1": 1.0,
                "num_correct_predictions": 10,
                "num_predictions": 10,
                "true_positives": 5,
                "true_negatives": 5,
                "false_positives": 0,
                "false_negatives": 0,
            },
            id="all-correct",
        ),
    ],
)
def test_binary_confusion_matrix_result_to_dict(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    betas: list[float],
    expected: dict,
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        betas=betas,
    )
    assert objects_are_allclose(m.to_dict(), expected)


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        pytest.param("train_", "", id="prefix-only"),
        pytest.param("", "_val", id="suffix-only"),
        pytest.param("train_", "_val", id="prefix-and-suffix"),
    ],
)
def test_binary_confusion_matrix_result_to_dict_prefix_suffix(prefix: str, suffix: str) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert objects_are_allclose(
        m.to_dict(prefix=prefix, suffix=suffix),
        {
            f"{prefix}accuracy{suffix}": 0.7,
            f"{prefix}precision{suffix}": 0.75,
            f"{prefix}recall{suffix}": 0.6,
            f"{prefix}specificity{suffix}": 0.8,
            f"{prefix}f1{suffix}": 0.6666666666666665,
            f"{prefix}num_correct_predictions{suffix}": 7,
            f"{prefix}num_predictions{suffix}": 10,
            f"{prefix}true_positives{suffix}": 3,
            f"{prefix}true_negatives{suffix}": 4,
            f"{prefix}false_positives{suffix}": 1,
            f"{prefix}false_negatives{suffix}": 2,
        },
    )


def test_binary_confusion_matrix_result_to_dict_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert objects_are_allclose(
        m.to_dict(),
        {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "num_correct_predictions": 0,
            "num_predictions": 0,
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        },
        equal_nan=True,
    )


# --- to_str ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "betas", "expected"),
    [
        pytest.param(
            3,
            4,
            1,
            2,
            [1.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=10  TP=3  TN=4  FP=1  FN=2\n"
                "Accuracy    [██████████████░░░░░░]  0.7000  (7/10)\n"
                "Precision   [███████████████░░░░░]  0.7500  (3/4)\n"
                "Recall      [████████████░░░░░░░░]  0.6000  (3/5)\n"
                "Specificity [████████████████░░░░]  0.8000  (4/5)\n"
                "F1          [█████████████░░░░░░░]  0.6667"
            ),
            id="standard",
        ),
        pytest.param(
            3,
            4,
            1,
            2,
            [0.5, 1.0, 2.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=10  TP=3  TN=4  FP=1  FN=2\n"
                "Accuracy    [██████████████░░░░░░]  0.7000  (7/10)\n"
                "Precision   [███████████████░░░░░]  0.7500  (3/4)\n"
                "Recall      [████████████░░░░░░░░]  0.6000  (3/5)\n"
                "Specificity [████████████████░░░░]  0.8000  (4/5)\n"
                "F0.5        [███████████████░░░░░]  0.7143\n"
                "F1          [█████████████░░░░░░░]  0.6667\n"
                "F2          [████████████░░░░░░░░]  0.6250"
            ),
            id="multiple-betas",
        ),
        pytest.param(
            5,
            5,
            0,
            0,
            [1.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=10  TP=5  TN=5  FP=0  FN=0\n"
                "Accuracy    [████████████████████]  1.0000  (10/10)\n"
                "Precision   [████████████████████]  1.0000  (5/5)\n"
                "Recall      [████████████████████]  1.0000  (5/5)\n"
                "Specificity [████████████████████]  1.0000  (5/5)\n"
                "F1          [████████████████████]  1.0000"
            ),
            id="all-correct",
        ),
        pytest.param(
            0,
            0,
            5,
            5,
            [1.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=10  TP=0  TN=0  FP=5  FN=5\n"
                "Accuracy    [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/10)\n"
                "Precision   [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/5)\n"
                "Recall      [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/5)\n"
                "Specificity [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/5)\n"
                "F1          [░░░░░░░░░░░░░░░░░░░░]  0.0000"
            ),
            id="all-incorrect",
        ),
        pytest.param(
            10,
            0,
            0,
            0,
            [1.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=10  TP=10  TN=0  FP=0  FN=0\n"
                "Accuracy    [████████████████████]  1.0000  (10/10)\n"
                "Precision   [████████████████████]  1.0000  (10/10)\n"
                "Recall      [████████████████████]  1.0000  (10/10)\n"
                "Specificity [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/0)\n"
                "F1          [████████████████████]  1.0000"
            ),
            id="all-tp",
        ),
        pytest.param(
            0,
            10,
            0,
            0,
            [1.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=10  TP=0  TN=10  FP=0  FN=0\n"
                "Accuracy    [████████████████████]  1.0000  (10/10)\n"
                "Precision   [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/0)\n"
                "Recall      [░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/0)\n"
                "Specificity [████████████████████]  1.0000  (10/10)\n"
                "F1          [░░░░░░░░░░░░░░░░░░░░]  0.0000"
            ),
            id="all-tn",
        ),
        pytest.param(
            1000,
            2000,
            500,
            500,
            [1.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=4,000  TP=1,000  TN=2,000  FP=500  FN=500\n"
                "Accuracy    [████████████████████]  0.7500  (3,000/4,000)\n"
                "Precision   [██████████████████░░]  0.6667  (1,000/1,500)\n"
                "Recall      [██████████████████░░]  0.6667  (1,000/1,500)\n"
                "Specificity [████████████████████]  0.8000  (2,000/2,500)\n"
                "F1          [██████████████████░░]  0.6667"
            ),
            id="large-numbers-thousands-separator",
        ),
        pytest.param(
            0,
            0,
            0,
            0,
            [1.0],
            (
                "Binary Confusion Matrix\n"
                "-----------------------\n"
                "n=0  TP=0  TN=0  FP=0  FN=0\n"
                "Accuracy    [░░░░░░░░░░░░░░░░░░░░]  nan  (0/0)\n"
                "Precision   [░░░░░░░░░░░░░░░░░░░░]  nan  (0/0)\n"
                "Recall      [░░░░░░░░░░░░░░░░░░░░]  nan  (0/0)\n"
                "Specificity [░░░░░░░░░░░░░░░░░░░░]  nan  (0/0)\n"
                "F1          [░░░░░░░░░░░░░░░░░░░░]  nan"
            ),
            id="zero-predictions",
        ),
    ],
)
def test_to_str(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    betas: list[float],
    expected: str,
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        betas=betas,
    )
    assert m.to_str() == expected


# --- from_confusion_matrix ---


def test_binary_confusion_matrix_result_from_confusion_matrix() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2
    )
    assert m.true_positives == 3
    assert m.true_negatives == 4
    assert m.false_positives == 1
    assert m.false_negatives == 2
    assert m.num_predictions == 10
    assert m.num_correct_predictions == 7
    assert m.accuracy == 0.7
    assert m.precision == 0.75
    assert m.recall == 0.6
    assert m.specificity == 0.8
    assert m.f_beta_scores == {1.0: 0.6666666666666665}


@pytest.mark.parametrize(
    "betas",
    [
        pytest.param((1.0,), id="tuple"),
        pytest.param([1.0], id="list"),
        pytest.param([0.5, 1.0, 2.0], id="list-multiple"),
        pytest.param((0.5, 1.0, 2.0), id="tuple-multiple"),
    ],
)
def test_binary_confusion_matrix_result_from_confusion_matrix_betas_sequence(
    betas: list[float] | tuple[float, ...],
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3, true_negatives=4, false_positives=1, false_negatives=2, betas=betas
    )
    assert set(m.f_beta_scores.keys()) == set(betas)


def test_binary_confusion_matrix_result_from_confusion_matrix_multiple_betas() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=3,
        true_negatives=4,
        false_positives=1,
        false_negatives=2,
        betas=[0.5, 1.0, 2.0],
    )
    assert m.f_beta_scores == {0.5: 0.7142857142857143, 1.0: 0.6666666666666665, 2.0: 0.625}


def test_binary_confusion_matrix_result_from_confusion_matrix_frozen() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
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
def test_binary_confusion_matrix_result_from_confusion_matrix_valid(
    tp: int, tn: int, fp: int, fn: int
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.true_positives == tp
    assert m.true_negatives == tn
    assert m.false_positives == fp
    assert m.false_negatives == fn


# --- Validation ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "match"),
    [
        pytest.param(-1, 4, 1, 2, "true_positives", id="negative-tp"),
        pytest.param(3, -1, 1, 2, "true_negatives", id="negative-tn"),
        pytest.param(3, 4, -1, 2, "false_positives", id="negative-fp"),
        pytest.param(3, 4, 1, -1, "false_negatives", id="negative-fn"),
    ],
)
def test_binary_confusion_matrix_result_negative_count_raises(
    tp: int, tn: int, fp: int, fn: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
        )


def test_binary_confusion_matrix_result_negative_beta_raises() -> None:
    with pytest.raises(ValueError, match="beta values must be >= 0"):
        BinaryConfusionMatrixResult.from_confusion_matrix(
            true_positives=3,
            true_negatives=4,
            false_positives=1,
            false_negatives=2,
            betas=[-1.0],
        )


# --- num_predictions ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 10, id="standard"),
        pytest.param(0, 0, 0, 0, 0, id="all-zero"),
        pytest.param(5, 0, 0, 0, 5, id="only-tp"),
        pytest.param(0, 0, 3, 7, 10, id="only-incorrect"),
    ],
)
def test_binary_confusion_matrix_result_num_predictions(
    tp: int, tn: int, fp: int, fn: int, expected: int
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.num_predictions == expected


# --- num_correct_predictions ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 7, id="standard"),
        pytest.param(0, 0, 0, 0, 0, id="all-zero"),
        pytest.param(5, 0, 0, 0, 5, id="only-tp"),
        pytest.param(0, 5, 0, 0, 5, id="only-tn"),
        pytest.param(0, 0, 5, 3, 0, id="no-correct"),
    ],
)
def test_binary_confusion_matrix_result_num_correct_predictions(
    tp: int, tn: int, fp: int, fn: int, expected: int
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.num_correct_predictions == expected


# --- accuracy ---


@pytest.mark.parametrize(
    ("tp", "tn", "fp", "fn", "expected"),
    [
        pytest.param(3, 4, 1, 2, 0.7, id="standard"),
        pytest.param(10, 0, 0, 0, 1.0, id="all-tp"),
        pytest.param(0, 10, 0, 0, 1.0, id="all-tn"),
        pytest.param(0, 0, 5, 5, 0.0, id="all-incorrect"),
        pytest.param(1, 1, 1, 1, 0.5, id="equal-counts"),
    ],
)
def test_binary_confusion_matrix_result_accuracy(
    tp: int, tn: int, fp: int, fn: int, expected: float
) -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=tp, true_negatives=tn, false_positives=fp, false_negatives=fn
    )
    assert m.accuracy == (expected)


def test_binary_confusion_matrix_result_accuracy_zero_predictions() -> None:
    m = BinaryConfusionMatrixResult.from_confusion_matrix(
        true_positives=0, true_negatives=0, false_positives=0, false_negatives=0
    )
    assert math.isnan(m.accuracy)
