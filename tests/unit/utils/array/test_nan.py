from __future__ import annotations

import numpy as np
import pytest

from mlev.utils.array import NAN_POLICIES, NanPolicy, check_nan_policy, contains_nan

######################################
#     Tests for check_nan_policy     #
######################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_check_nan_policy_valid(nan_policy: str) -> None:
    check_nan_policy(nan_policy)


def test_check_nan_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        check_nan_policy("incorrect")


##################################
#     Tests for contains_nan     #
##################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_contains_nan_no_nan(nan_policy: NanPolicy) -> None:
    assert not contains_nan(np.array([1, 2, 3, 4, 5]), nan_policy=nan_policy)


def test_contains_nan_omit() -> None:
    assert contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="omit")


def test_contains_nan_propagate() -> None:
    assert contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="propagate")


def test_contains_nan_raise() -> None:
    with pytest.raises(ValueError, match="input contains at least one NaN value"):
        contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="raise")


def test_contains_nan_raise_name() -> None:
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="raise", name="'x'")


# --- Object dtype ---


def test_contains_nan_object_no_nan() -> None:
    assert not contains_nan(np.array([1, 2, 3], dtype=object))


def test_contains_nan_object_with_nan() -> None:
    assert contains_nan(np.array([1, float("nan"), 3], dtype=object))


def test_contains_nan_object_with_none_only() -> None:
    # None is not NaN, so should return False
    assert not contains_nan(np.array([1, None, 3], dtype=object))


def test_contains_nan_object_with_nan_and_none() -> None:
    assert contains_nan(np.array([None, float("nan"), 1], dtype=object))


def test_contains_nan_object_raise() -> None:
    with pytest.raises(ValueError, match="input contains at least one NaN value"):
        contains_nan(np.array([1, float("nan")], dtype=object), nan_policy="raise")


# --- Non-numeric dtypes ---


def test_contains_nan_str_array() -> None:
    assert not contains_nan(np.array(["a", "b", "c"]))


def test_contains_nan_datetime_array() -> None:
    assert not contains_nan(np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"))


# --- Multidimensional ---


def test_contains_nan_2d_with_nan() -> None:
    assert contains_nan(np.array([[1.0, np.nan], [3.0, 4.0]]))


def test_contains_nan_2d_no_nan() -> None:
    assert not contains_nan(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_contains_nan_2d_object_with_nan() -> None:
    assert contains_nan(np.array([[1, float("nan")], [3, 4]], dtype=object))


# --- Edge cases ---


def test_contains_nan_empty_array() -> None:
    assert not contains_nan(np.array([], dtype=float))


def test_contains_nan_empty_object_array() -> None:
    assert not contains_nan(np.array([], dtype=object))


def test_contains_nan_all_nan() -> None:
    assert contains_nan(np.array([np.nan, np.nan, np.nan]))
