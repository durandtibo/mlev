from __future__ import annotations

import numpy as np
import pytest
from coola.equality import objects_are_equal

from mlev.utils.array import preprocess

##################################
#     Tests for preprocess       #
##################################


# --- drop_missing=False (default) ---


def test_preprocess_no_missing_keep() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert objects_are_equal(preprocess(arrays), arrays)


def test_preprocess_with_missing_keep() -> None:
    arrays = [np.array([1.0, np.nan, 3.0]), np.array([4.0, 5.0, np.nan])]
    assert objects_are_equal(preprocess(arrays), arrays, equal_nan=True)


def test_preprocess_returns_list_keep() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert isinstance(preprocess(arrays), list)


def test_preprocess_single_array_keep() -> None:
    arrays = [np.array([1.0, np.nan, 3.0])]
    assert objects_are_equal(preprocess(arrays), arrays, equal_nan=True)


def test_preprocess_three_arrays_keep() -> None:
    arrays = [
        np.array([1.0, np.nan, 3.0]),
        np.array([4.0, 5.0, np.nan]),
        np.array([7.0, 8.0, 9.0]),
    ]
    assert objects_are_equal(preprocess(arrays), arrays, equal_nan=True)


# --- drop_missing=True ---


def test_preprocess_no_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            drop_missing=True,
        ),
        [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
    )


def test_preprocess_missing_in_first_array_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [np.array([1.0, np.nan, 3.0]), np.array([4.0, 5.0, 6.0])],
            drop_missing=True,
        ),
        [np.array([1.0, 3.0]), np.array([4.0, 6.0])],
    )


def test_preprocess_missing_in_last_array_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, np.nan, 6.0])],
            drop_missing=True,
        ),
        [np.array([1.0, 3.0]), np.array([4.0, 6.0])],
    )


def test_preprocess_missing_in_all_arrays_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                np.array([1.0, np.nan, 3.0]),
                np.array([4.0, 5.0, np.nan]),
                np.array([np.nan, 8.0, 9.0]),
            ],
            drop_missing=True,
        ),
        [np.array([]), np.array([]), np.array([])],
    )


def test_preprocess_missing_overlap_drop() -> None:
    # Both arrays have NaN at the same position — only one row dropped
    assert objects_are_equal(
        preprocess(
            [np.array([1.0, np.nan, 3.0]), np.array([4.0, np.nan, 6.0])],
            drop_missing=True,
        ),
        [np.array([1.0, 3.0]), np.array([4.0, 6.0])],
    )


def test_preprocess_missing_in_both_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan]),
                np.array([1.0, np.nan, 0.0, 1.0, 0.0, np.nan]),
            ],
            drop_missing=True,
        ),
        [np.array([1.0, 4.0, 5.0]), np.array([1.0, 1.0, 0.0])],
    )


def test_preprocess_all_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [np.array([np.nan, np.nan]), np.array([np.nan, np.nan])],
            drop_missing=True,
        ),
        [np.array([]), np.array([])],
    )


def test_preprocess_single_array_drop() -> None:
    assert objects_are_equal(
        preprocess([np.array([1.0, np.nan, 3.0])], drop_missing=True),
        [np.array([1.0, 3.0])],
    )


def test_preprocess_three_arrays_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                np.array([1.0, np.nan, 3.0]),
                np.array([4.0, 5.0, 6.0]),
                np.array([7.0, 8.0, 9.0]),
            ],
            drop_missing=True,
        ),
        [np.array([1.0, 3.0]), np.array([4.0, 6.0]), np.array([7.0, 9.0])],
    )


# --- Object arrays with None ---


def test_preprocess_object_none_in_first_array_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [np.array([1, None, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            drop_missing=True,
        ),
        [np.array([1, 3], dtype=object), np.array([4, 6], dtype=object)],
    )


def test_preprocess_object_none_in_second_array_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [np.array([1, 2, 3], dtype=object), np.array([4, None, 6], dtype=object)],
            drop_missing=True,
        ),
        [np.array([1, 3], dtype=object), np.array([4, 6], dtype=object)],
    )


def test_preprocess_object_none_and_nan_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                np.array([1, None, 3], dtype=object),
                np.array([float("nan"), 2, 3], dtype=object),
            ],
            drop_missing=True,
        ),
        [np.array([3], dtype=object), np.array([3], dtype=object)],
    )


def test_preprocess_object_keep() -> None:
    arrays = [np.array([1, None, 3], dtype=object), np.array([4, 5, None], dtype=object)]
    assert objects_are_equal(preprocess(arrays), arrays)


# --- Output length matches input length ---


def test_preprocess_output_length_matches_input_keep() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])]
    assert len(preprocess(arrays)) == len(arrays)


def test_preprocess_output_length_matches_input_drop() -> None:
    arrays = [np.array([1.0, np.nan, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])]
    assert len(preprocess(arrays, drop_missing=True)) == len(arrays)


# --- Returns list ---


def test_preprocess_returns_list_from_tuple() -> None:
    arrays = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert isinstance(preprocess(arrays), list)


def test_preprocess_returns_list_drop() -> None:
    arrays = [np.array([1.0, np.nan, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert isinstance(preprocess(arrays, drop_missing=True), list)


# --- Shape mismatch ---


def test_preprocess_different_shapes_raises() -> None:
    with pytest.raises(ValueError, match="arrays have different shapes:"):
        preprocess([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])])


def test_preprocess_different_shapes_drop_raises() -> None:
    with pytest.raises(ValueError, match="arrays have different shapes:"):
        preprocess(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])],
            drop_missing=True,
        )


# --- Empty input raises ---


def test_preprocess_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="'arrays' cannot be empty"):
        preprocess([])


# --- Edge cases ---


def test_preprocess_empty_arrays_keep() -> None:
    arrays = [np.array([], dtype=float), np.array([], dtype=float)]
    assert objects_are_equal(preprocess(arrays), arrays)


def test_preprocess_empty_arrays_drop() -> None:
    assert objects_are_equal(
        preprocess([np.array([], dtype=float), np.array([], dtype=float)], drop_missing=True),
        [np.array([], dtype=float), np.array([], dtype=float)],
    )


def test_preprocess_single_element_no_missing_drop() -> None:
    assert objects_are_equal(
        preprocess([np.array([1.0]), np.array([2.0])], drop_missing=True),
        [np.array([1.0]), np.array([2.0])],
    )


def test_preprocess_single_element_missing_drop() -> None:
    assert objects_are_equal(
        preprocess([np.array([np.nan]), np.array([1.0])], drop_missing=True),
        [np.array([]), np.array([])],
    )
