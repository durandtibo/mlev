from __future__ import annotations

import numpy as np
import pytest
from coola.equality import objects_are_equal

from mlev.utils.array import (
    contains_missing,
    contains_none,
    is_missing,
    multi_is_missing,
)

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


class BadEqTypeError:
    """Element whose __eq__ raises TypeError, triggering the
    fallback."""

    def __eq__(self, other: object) -> bool:
        msg = "cannot compare"
        raise TypeError(msg)


def test_contains_none_fallback_with_none() -> None:
    arr = np.array([BadEq(), None], dtype=object)
    assert contains_none(arr)


def test_contains_none_fallback_without_none() -> None:
    arr = np.array([BadEq(), BadEq()], dtype=object)
    assert not contains_none(arr)


def test_contains_none_fallback_type_error_with_none() -> None:
    arr = np.array([BadEqTypeError(), None], dtype=object)
    assert contains_none(arr)


def test_contains_none_fallback_type_error_without_none() -> None:
    arr = np.array([BadEqTypeError(), BadEqTypeError()], dtype=object)
    assert not contains_none(arr)


##################################
#     Tests for is_missing       #
##################################


# --- Float arrays ---


def test_is_missing_float_no_missing() -> None:
    result = is_missing(np.array([1.0, 2.0, 3.0]))
    assert objects_are_equal(result, np.array([False, False, False], dtype=bool))


def test_is_missing_float_with_nan() -> None:
    result = is_missing(np.array([1.0, float("nan"), 3.0]))
    assert objects_are_equal(result, np.array([False, True, False], dtype=bool))


def test_is_missing_float_all_nan() -> None:
    result = is_missing(np.array([float("nan"), float("nan")]))
    assert objects_are_equal(result, np.array([True, True], dtype=bool))


def test_is_missing_float_nan_at_start() -> None:
    result = is_missing(np.array([float("nan"), 2.0, 3.0]))
    assert objects_are_equal(result, np.array([True, False, False], dtype=bool))


def test_is_missing_float_nan_at_end() -> None:
    result = is_missing(np.array([1.0, 2.0, float("nan")]))
    assert objects_are_equal(result, np.array([False, False, True], dtype=bool))


# --- Object arrays ---


def test_is_missing_object_no_missing() -> None:
    result = is_missing(np.array([1, 2, 3], dtype=object))
    assert objects_are_equal(result, np.array([False, False, False], dtype=bool))


def test_is_missing_object_with_none() -> None:
    result = is_missing(np.array([1, None, 3], dtype=object))
    assert objects_are_equal(result, np.array([False, True, False], dtype=bool))


def test_is_missing_object_with_nan() -> None:
    result = is_missing(np.array([1, float("nan"), 3], dtype=object))
    assert objects_are_equal(result, np.array([False, True, False], dtype=bool))


def test_is_missing_object_with_nan_and_none() -> None:
    result = is_missing(np.array([None, float("nan"), 3], dtype=object))
    assert objects_are_equal(result, np.array([True, True, False], dtype=bool))


def test_is_missing_object_all_none() -> None:
    result = is_missing(np.array([None, None, None], dtype=object))
    assert objects_are_equal(result, np.array([True, True, True], dtype=bool))


def test_is_missing_object_none_not_confused_with_zero() -> None:
    result = is_missing(np.array([0, False, "", []], dtype=object))
    assert objects_are_equal(result, np.array([False, False, False, False], dtype=bool))


# --- Non-numeric dtypes ---


def test_is_missing_str_array() -> None:
    result = is_missing(np.array(["a", "b", "c"]))
    assert objects_are_equal(result, np.array([False, False, False], dtype=bool))


def test_is_missing_datetime_array() -> None:
    result = is_missing(np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"))
    assert objects_are_equal(result, np.array([False, False], dtype=bool))


def test_is_missing_bool_array() -> None:
    result = is_missing(np.array([True, False, True]))
    assert objects_are_equal(result, np.array([False, False, False], dtype=bool))


def test_is_missing_int_array() -> None:
    result = is_missing(np.array([1, 2, 3]))
    assert objects_are_equal(result, np.array([False, False, False], dtype=bool))


# --- Return dtype ---


def test_is_missing_returns_bool_dtype() -> None:
    assert is_missing(np.array([1.0, float("nan")])).dtype == bool


def test_is_missing_object_returns_bool_dtype() -> None:
    assert is_missing(np.array([1, None], dtype=object)).dtype == bool


# --- Multidimensional arrays ---


def test_is_missing_2d_float_with_nan() -> None:
    result = is_missing(np.array([[1.0, float("nan")], [3.0, 4.0]]))
    assert objects_are_equal(result, np.array([[False, True], [False, False]], dtype=bool))


def test_is_missing_2d_float_no_missing() -> None:
    result = is_missing(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert objects_are_equal(result, np.array([[False, False], [False, False]], dtype=bool))


def test_is_missing_2d_object_with_none() -> None:
    result = is_missing(np.array([[1, None], [3, 4]], dtype=object))
    assert objects_are_equal(result, np.array([[False, True], [False, False]], dtype=bool))


def test_is_missing_2d_preserves_shape() -> None:
    arr = np.array([[1.0, float("nan")], [3.0, 4.0]])
    assert is_missing(arr).shape == arr.shape


# --- Empty arrays ---


@pytest.mark.parametrize("dtype", [float, object, bool, int])
def test_is_missing_empty_array(dtype: type) -> None:
    result = is_missing(np.array([], dtype=dtype))
    assert objects_are_equal(result, np.array([], dtype=bool))


##################################
#   Tests for multi_is_missing   #
##################################


# --- Single array ---


def test_multi_is_missing_single_array_no_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0, 2.0, 3.0])]),
        np.array([False, False, False], dtype=bool),
    )


def test_multi_is_missing_single_array_with_nan() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0, float("nan"), 3.0])]),
        np.array([False, True, False], dtype=bool),
    )


def test_multi_is_missing_single_array_with_none() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1, None, 3], dtype=object)]),
        np.array([False, True, False], dtype=bool),
    )


def test_multi_is_missing_single_array_all_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([float("nan"), float("nan")])]),
        np.array([True, True], dtype=bool),
    )


# --- Multiple arrays ---


def test_multi_is_missing_two_arrays_no_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]),
        np.array([False, False, False], dtype=bool),
    )


def test_multi_is_missing_two_arrays_first_has_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])]),
        np.array([False, True, False], dtype=bool),
    )


def test_multi_is_missing_two_arrays_second_has_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0, 2.0, 3.0]), np.array([4.0, float("nan"), 6.0])]),
        np.array([False, True, False], dtype=bool),
    )


def test_multi_is_missing_two_arrays_both_have_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0, float("nan"), 3.0]), np.array([float("nan"), 5.0, 6.0])]),
        np.array([True, True, False], dtype=bool),
    )


def test_multi_is_missing_two_arrays_missing_overlap() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0, float("nan"), 3.0]), np.array([4.0, float("nan"), 6.0])]),
        np.array([False, True, False], dtype=bool),
    )


def test_multi_is_missing_three_arrays() -> None:
    assert objects_are_equal(
        multi_is_missing(
            [
                np.array([1.0, float("nan"), 3.0]),
                np.array([4.0, 5.0, float("nan")]),
                np.array([float("nan"), 8.0, 9.0]),
            ]
        ),
        np.array([True, True, True], dtype=bool),
    )


# --- Mixed object arrays with None and NaN ---


def test_multi_is_missing_object_none_and_nan() -> None:
    assert objects_are_equal(
        multi_is_missing(
            [
                np.array([1, None, 3], dtype=object),
                np.array([float("nan"), 2, 3], dtype=object),
            ]
        ),
        np.array([True, True, False], dtype=bool),
    )


def test_multi_is_missing_object_no_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1, 2, 3], dtype=object), np.array([4, 5, 6], dtype=object)]),
        np.array([False, False, False], dtype=bool),
    )


# --- Non-numeric dtypes ---


def test_multi_is_missing_str_arrays() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array(["a", "b", "c"]), np.array(["d", "e", "f"])]),
        np.array([False, False, False], dtype=bool),
    )


def test_multi_is_missing_datetime_arrays() -> None:
    assert objects_are_equal(
        multi_is_missing(
            [
                np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
                np.array(["2021-01-03", "2021-01-04"], dtype="datetime64"),
            ]
        ),
        np.array([False, False], dtype=bool),
    )


# --- Return dtype ---


def test_multi_is_missing_returns_bool_dtype() -> None:
    assert multi_is_missing([np.array([1.0, float("nan")])]).dtype == bool


def test_multi_is_missing_object_returns_bool_dtype() -> None:
    assert multi_is_missing([np.array([1, None], dtype=object)]).dtype == bool


# --- Multidimensional arrays ---


def test_multi_is_missing_2d_with_nan() -> None:
    assert objects_are_equal(
        multi_is_missing(
            [
                np.array([[1.0, float("nan")], [3.0, 4.0]]),
                np.array([[float("nan"), 2.0], [3.0, 4.0]]),
            ]
        ),
        np.array([[True, True], [False, False]], dtype=bool),
    )


def test_multi_is_missing_2d_preserves_shape() -> None:
    arrays = [
        np.array([[1.0, float("nan")], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [float("nan"), 8.0]]),
    ]
    assert multi_is_missing(arrays).shape == arrays[0].shape


# --- Edge cases ---


def test_multi_is_missing_empty_raises() -> None:
    with pytest.raises(ValueError, match="'arrays' cannot be empty"):
        multi_is_missing([])


def test_multi_is_missing_single_element_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([float("nan")])]),
        np.array([True], dtype=bool),
    )


def test_multi_is_missing_single_element_no_missing() -> None:
    assert objects_are_equal(
        multi_is_missing([np.array([1.0])]),
        np.array([False], dtype=bool),
    )
