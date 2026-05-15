from __future__ import annotations

import numpy as np
import pytest

from mlev.utils.array import NAN_POLICIES, check_nan_policy, contains_nan

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
def test_contains_nan_no_nan(nan_policy: str) -> None:
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
