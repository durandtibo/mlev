from __future__ import annotations

import numpy as np
import pytest

from mlev.utils.array import check_array_ndim

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
