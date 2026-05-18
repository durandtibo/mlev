from __future__ import annotations

import polars as pl
import pytest
from coola.equality import objects_are_equal

from mlev.utils.series import preprocess

################################
#     Tests for preprocess     #
################################


# --- drop_missing=False (default) ---


def test_preprocess_no_missing_keep() -> None:
    series = [pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])]
    assert objects_are_equal(preprocess(series), series)


def test_preprocess_with_missing_keep() -> None:
    series = [
        pl.Series("y_true", [1, 0, 0, 1, 1, None]),
        pl.Series("y_pred", [0, 1, 0, 1, None, 1]),
    ]
    assert objects_are_equal(preprocess(series), series)


def test_preprocess_returns_list_keep() -> None:
    series = [pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])]
    assert isinstance(preprocess(series), list)


def test_preprocess_single_series_keep() -> None:
    series = [pl.Series("y_true", [1, 0, None, 1])]
    assert objects_are_equal(preprocess(series), series)


def test_preprocess_three_series_keep() -> None:
    series = [
        pl.Series("a", [1, 0, None, 1]),
        pl.Series("b", [0, 1, 0, None]),
        pl.Series("c", [1, 1, 1, 1]),
    ]
    assert objects_are_equal(preprocess(series), series)


# --- drop_missing=True ---


def test_preprocess_no_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])],
            drop_missing=True,
        ),
        [pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])],
    )


def test_preprocess_missing_in_first_series_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [pl.Series("y_true", [1, None, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])],
            drop_missing=True,
        ),
        [pl.Series("y_true", [1, 0, 1]), pl.Series("y_pred", [0, 0, 1])],
    )


def test_preprocess_missing_in_last_series_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, None, 0, 1])],
            drop_missing=True,
        ),
        [pl.Series("y_true", [1, 0, 1]), pl.Series("y_pred", [0, 0, 1])],
    )


def test_preprocess_missing_in_both_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                pl.Series("y_true", [1, 0, 0, 1, 1, None]),
                pl.Series("y_pred", [0, 1, 0, 1, None, 1]),
            ],
            drop_missing=True,
        ),
        [pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])],
    )


def test_preprocess_missing_overlap_drop() -> None:
    # Both series have null at the same position — only one row dropped
    assert objects_are_equal(
        preprocess(
            [pl.Series("y_true", [1, None, 0, 1]), pl.Series("y_pred", [0, None, 0, 1])],
            drop_missing=True,
        ),
        [pl.Series("y_true", [1, 0, 1]), pl.Series("y_pred", [0, 0, 1])],
    )


def test_preprocess_missing_in_all_series_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                pl.Series("a", [1, None, 3]),
                pl.Series("b", [4, 5, None]),
                pl.Series("c", [None, 8, 9]),
            ],
            drop_missing=True,
        ),
        [
            pl.Series("a", [], dtype=pl.Int64),
            pl.Series("b", [], dtype=pl.Int64),
            pl.Series("c", [], dtype=pl.Int64),
        ],
    )


def test_preprocess_all_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                pl.Series("y_true", [None, None], dtype=pl.Int64),
                pl.Series("y_pred", [None, None], dtype=pl.Int64),
            ],
            drop_missing=True,
        ),
        [
            pl.Series("y_true", [], dtype=pl.Int64),
            pl.Series("y_pred", [], dtype=pl.Int64),
        ],
    )


def test_preprocess_single_series_drop() -> None:
    assert objects_are_equal(
        preprocess([pl.Series("x", [1, None, 3])], drop_missing=True),
        [pl.Series("x", [1, 3])],
    )


def test_preprocess_three_series_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [
                pl.Series("a", [1, None, 3]),
                pl.Series("b", [4, 5, 6]),
                pl.Series("c", [7, 8, 9]),
            ],
            drop_missing=True,
        ),
        [pl.Series("a", [1, 3]), pl.Series("b", [4, 6]), pl.Series("c", [7, 9])],
    )


# --- Series names are preserved ---


def test_preprocess_preserves_series_names_keep() -> None:
    result = preprocess([pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])])
    assert result[0].name == "y_true"
    assert result[1].name == "y_pred"


def test_preprocess_preserves_series_names_drop() -> None:
    result = preprocess(
        [pl.Series("y_true", [1, None, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1])],
        drop_missing=True,
    )
    assert result[0].name == "y_true"
    assert result[1].name == "y_pred"


# --- Output length matches input length ---


def test_preprocess_output_length_matches_input_keep() -> None:
    series = [pl.Series("a", [1, 2, 3]), pl.Series("b", [4, 5, 6]), pl.Series("c", [7, 8, 9])]
    assert len(preprocess(series)) == len(series)


def test_preprocess_output_length_matches_input_drop() -> None:
    series = [pl.Series("a", [1, None, 3]), pl.Series("b", [4, 5, 6]), pl.Series("c", [7, 8, 9])]
    assert len(preprocess(series, drop_missing=True)) == len(series)


# --- Returns list ---


def test_preprocess_returns_list_from_tuple() -> None:
    series = (pl.Series("y_true", [1, 0, 0, 1]), pl.Series("y_pred", [0, 1, 0, 1]))
    assert isinstance(preprocess(series), list)


def test_preprocess_returns_list_drop() -> None:
    series = [pl.Series("y_true", [1, None, 3]), pl.Series("y_pred", [4, 5, 6])]
    assert isinstance(preprocess(series, drop_missing=True), list)


# --- Different dtypes ---


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(
            [
                pl.Series("a", [1, None, 0], dtype=pl.Int32),
                pl.Series("b", [0, 1, None], dtype=pl.Int32),
            ],
            id="int32",
        ),
        pytest.param(
            [
                pl.Series("a", [1.0, None, 0.0], dtype=pl.Float32),
                pl.Series("b", [0.0, 1.0, None], dtype=pl.Float32),
            ],
            id="float32",
        ),
        pytest.param(
            [
                pl.Series("a", [1.0, None, 0.0], dtype=pl.Float64),
                pl.Series("b", [0.0, 1.0, None], dtype=pl.Float64),
            ],
            id="float64",
        ),
        pytest.param(
            [
                pl.Series("a", ["x", None, "z"], dtype=pl.String),
                pl.Series("b", ["d", "e", None], dtype=pl.String),
            ],
            id="str",
        ),
        pytest.param(
            [
                pl.Series("a", [True, None, False], dtype=pl.Boolean),
                pl.Series("b", [False, True, None], dtype=pl.Boolean),
            ],
            id="bool",
        ),
    ],
)
def test_preprocess_drop_missing_dtypes(series: list[pl.Series]) -> None:
    result = preprocess(series, drop_missing=True)
    assert all(s.null_count() == 0 for s in result)
    assert all(len(s) == 1 for s in result)


# --- Shape mismatch ---


def test_preprocess_different_shapes_raises() -> None:
    with pytest.raises(ValueError, match="series have different shapes:"):
        preprocess([pl.Series("y_true", [1, 0, 0]), pl.Series("y_pred", [0, 1])])


def test_preprocess_different_shapes_drop_raises() -> None:
    with pytest.raises(ValueError, match="series have different shapes:"):
        preprocess(
            [pl.Series("y_true", [1, 0, 0]), pl.Series("y_pred", [0, 1])],
            drop_missing=True,
        )


# --- Empty input returns empty list ---


def test_preprocess_empty_input_returns_empty_list() -> None:
    assert preprocess([]) == []


def test_preprocess_empty_input_drop_returns_empty_list() -> None:
    assert preprocess([], drop_missing=True) == []


# --- Edge cases ---


def test_preprocess_empty_series_keep() -> None:
    series = [pl.Series("y_true", [], dtype=pl.Int64), pl.Series("y_pred", [], dtype=pl.Int64)]
    assert objects_are_equal(preprocess(series), series)


def test_preprocess_empty_series_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [pl.Series("y_true", [], dtype=pl.Int64), pl.Series("y_pred", [], dtype=pl.Int64)],
            drop_missing=True,
        ),
        [pl.Series("y_true", [], dtype=pl.Int64), pl.Series("y_pred", [], dtype=pl.Int64)],
    )


def test_preprocess_single_element_no_missing_drop() -> None:
    assert objects_are_equal(
        preprocess([pl.Series("y_true", [1]), pl.Series("y_pred", [0])], drop_missing=True),
        [pl.Series("y_true", [1]), pl.Series("y_pred", [0])],
    )


def test_preprocess_single_element_missing_drop() -> None:
    assert objects_are_equal(
        preprocess(
            [pl.Series("y_true", [None], dtype=pl.Int64), pl.Series("y_pred", [1])],
            drop_missing=True,
        ),
        [pl.Series("y_true", [], dtype=pl.Int64), pl.Series("y_pred", [], dtype=pl.Int64)],
    )
