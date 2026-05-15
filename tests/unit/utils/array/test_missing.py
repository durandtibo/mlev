from __future__ import annotations

import numpy as np
import pytest

from mlev.utils.array import contains_missing, contains_none

######################################
#     Tests for contains_missing     #
######################################

# --- Basic cases ---


def test_contains_missing_with_nan() -> None:
    arr = np.array([1.0, 2.0, np.nan])
    assert contains_missing(arr)


def test_contains_missing_without_nan() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert not contains_missing(arr)


def test_contains_missing_with_none() -> None:
    arr = np.array([1, 2, None], dtype=object)
    assert contains_missing(arr)


def test_contains_missing_without_none() -> None:
    arr = np.array([1, 2, 3], dtype=object)
    assert not contains_missing(arr)


# --- Object dtype with np.nan ---


def test_contains_missing_object_array_with_nan() -> None:
    arr = np.array([1, float("nan")], dtype=object)
    assert contains_missing(arr)


def test_contains_missing_object_array_with_nan_and_none() -> None:
    arr = np.array([None, float("nan"), 1], dtype=object)
    assert contains_missing(arr)


def test_contains_missing_object_array_no_missing() -> None:
    arr = np.array([1, 2, 3], dtype=object)
    assert not contains_missing(arr)


# --- Non-numeric dtypes ---


def test_contains_missing_datetime_array_no_missing() -> None:
    arr = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64")
    assert not contains_missing(arr)


def test_contains_missing_str_array_no_missing() -> None:
    arr = np.array(["a", "b", "c"])
    assert not contains_missing(arr)


# --- missing_policy='propagate' (default) ---


def test_contains_missing_propagate_returns_true_on_missing() -> None:
    arr = np.array([1.0, np.nan])
    assert contains_missing(arr, missing_policy="propagate") is True


def test_contains_missing_propagate_returns_false_on_no_missing() -> None:
    arr = np.array([1.0, 2.0])
    assert contains_missing(arr, missing_policy="propagate") is False


# --- missing_policy='omit' ---


def test_contains_missing_omit_returns_true_on_missing() -> None:
    arr = np.array([1.0, np.nan])
    assert contains_missing(arr, missing_policy="omit") is True


def test_contains_missing_omit_returns_false_on_no_missing() -> None:
    arr = np.array([1.0, 2.0])
    assert contains_missing(arr, missing_policy="omit") is False


# --- missing_policy='raise' ---


def test_contains_missing_raise_raises_on_missing() -> None:
    arr = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="input contains at least one missing value"):
        contains_missing(arr, missing_policy="raise")


def test_contains_missing_raise_custom_name() -> None:
    arr = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="my_array contains at least one missing value"):
        contains_missing(arr, missing_policy="raise", name="my_array")


def test_contains_missing_raise_no_missing_does_not_raise() -> None:
    arr = np.array([1.0, 2.0])
    assert contains_missing(arr, missing_policy="raise") is False


# --- Multidimensional ---


def test_contains_missing_2d_array_with_nan() -> None:
    arr = np.array([[1.0, np.nan], [3.0, 4.0]])
    assert contains_missing(arr)


def test_contains_missing_2d_object_array_with_none() -> None:
    arr = np.array([[1, None], [3, 4]], dtype=object)
    assert contains_missing(arr)


# --- Edge cases ---


def test_contains_missing_empty_array() -> None:
    arr = np.array([], dtype=float)
    assert not contains_missing(arr)


def test_contains_missing_empty_object_array() -> None:
    arr = np.array([], dtype=object)
    assert not contains_missing(arr)


###################################
#     Tests for contains_none     #
###################################

# --- Basic cases ---


def test_contains_none_with_none() -> None:
    arr = np.array([1, 2, None, 4], dtype=object)
    assert contains_none(arr)


def test_contains_none_without_none() -> None:
    arr = np.array([1, 2, 3, 4], dtype=object)
    assert not contains_none(arr)


def test_contains_none_all_none() -> None:
    arr = np.array([None, None, None], dtype=object)
    assert contains_none(arr)


def test_contains_none_only_element_is_none() -> None:
    arr = np.array([None], dtype=object)
    assert contains_none(arr)


def test_contains_none_none_at_start() -> None:
    arr = np.array([None, 1, 2], dtype=object)
    assert contains_none(arr)


def test_contains_none_none_at_end() -> None:
    arr = np.array([1, 2, None], dtype=object)
    assert contains_none(arr)


# --- Multidimensional ---


def test_contains_none_2d_contains_none() -> None:
    arr = np.array([[1, None], [3, 4]], dtype=object)
    assert contains_none(arr)


def test_contains_none_2d_no_none() -> None:
    arr = np.array([[1, 2], [3, 4]], dtype=object)
    assert not contains_none(arr)


# --- Non-object dtypes (no None values) ---


def test_contains_none_float_array_no_none() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert not contains_none(arr)


def test_contains_none_float_array_with_nan() -> None:
    arr = np.array([1.0, float("nan"), 3.0])
    assert not contains_none(arr)


def test_contains_none_int_array_no_none() -> None:
    arr = np.array([1, 2, 3])
    assert not contains_none(arr)


def test_contains_none_bool_array_no_none() -> None:
    arr = np.array([True, False, True])
    assert not contains_none(arr)


def test_contains_none_str_array_no_none() -> None:
    arr = np.array(["a", "b", "c"])
    assert not contains_none(arr)


def test_contains_none_2d_float_array_no_none() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert not contains_none(arr)


# --- Edge cases ---


def test_contains_none_empty_array() -> None:
    arr = np.array([], dtype=object)
    assert not contains_none(arr)


def test_contains_none_none_not_confused_with_nan() -> None:
    arr = np.array([1.0, float("nan"), 3.0], dtype=object)
    assert not contains_none(arr)


def test_contains_none_none_not_confused_with_zero() -> None:
    arr = np.array([0, False, "", []], dtype=object)
    assert not contains_none(arr)


# --- Fallback strategy (elements with non-scalar __eq__) ---


class BadEq:
    """Element whose __eq__ raises ValueError, triggering the
    fallback."""

    def __eq__(self, other: object) -> bool:
        msg = "cannot compare"
        raise ValueError(msg)


def test_contains_none_fallback_with_none() -> None:
    arr = np.array([BadEq(), None], dtype=object)
    assert contains_none(arr)


def test_contains_none_fallback_without_none() -> None:
    arr = np.array([BadEq(), BadEq()], dtype=object)
    assert not contains_none(arr)
