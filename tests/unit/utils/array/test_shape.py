from __future__ import annotations

import numpy as np
import pytest

from mlev.utils.array import check_array_ndim, check_same_shape

######################################
#     Tests for check_array_ndim     #
######################################


@pytest.mark.parametrize("shape", [(2,), (1,), (3,)])
def test_check_array_ndim_1(shape: tuple[int, ...]) -> None:
    check_array_ndim(arr=np.ones(shape), ndim=1)


@pytest.mark.parametrize("shape", [(2, 3), (1, 1), (3, 2)])
def test_check_array_ndim_2(shape: tuple[int, ...]) -> None:
    check_array_ndim(arr=np.ones(shape), ndim=2)


def test_check_array_ndim_incorrect() -> None:
    with pytest.raises(ValueError, match="input: expected 3D array"):
        check_array_ndim(np.ones((2, 3)), ndim=3)


def test_check_array_ndim_incorrect_custom_name() -> None:
    with pytest.raises(ValueError, match="predictions: expected 4D array"):
        check_array_ndim(np.ones((2, 3)), ndim=4, name="predictions")


######################################
#     Tests for check_same_shape     #
######################################


def test_check_same_shape_1_array() -> None:
    check_same_shape([np.array([1, 0, 0, 1, 1])])


def test_check_same_shape_2_arrays_correct() -> None:
    check_same_shape([np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4, 5])])


def test_check_same_shape_2_arrays_incorrect() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        check_same_shape([np.array([1, 0, 0, 1, 1]), np.array([1, 0, 0, 1])])


def test_check_same_shape_3_arrays_correct() -> None:
    check_same_shape(
        [np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1])]
    )


def test_check_same_shape_3_arrays_incorrect() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        check_same_shape(
            [np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4]), np.array([6, 5, 4, 3, 2, 1])]
        )
