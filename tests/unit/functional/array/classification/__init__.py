from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlev.functional.array.classification import accuracy
from mlev.results import AccuracyResult

##############################
#     Tests for accuracy     #
##############################


###################
# numpy arrays    #
###################


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
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4))


def test_accuracy_array_missing_in_y_pred_propagate() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, np.nan, 1.0]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4))


def test_accuracy_array_missing_in_both_propagate() -> None:
    assert accuracy(
        y_true=np.array([1.0, 0.0, np.nan, 1.0]),
        y_pred=np.array([1.0, np.nan, 0.0, 1.0]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4))


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
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0))


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
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0))


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


###################
# polars Series   #
###################


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
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4))


def test_accuracy_series_missing_in_y_pred_propagate() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, 0, 1]),
        y_pred=pl.Series("y_pred", [1, 0, None, 1]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4))


def test_accuracy_series_missing_in_both_propagate() -> None:
    assert accuracy(
        y_true=pl.Series("y_true", [1, 0, None, 1]),
        y_pred=pl.Series("y_pred", [1, None, 0, 1]),
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=4))


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
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0))


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
    ).equal(AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0))


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
