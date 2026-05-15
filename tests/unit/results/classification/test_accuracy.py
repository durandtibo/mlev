from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from coola.equality import objects_are_equal

from mlev.results.classification import AccuracyResult

####################################
#     Tests for AccuracyResult     #
####################################


def test_accuracy_result_num_correct_predictions() -> None:
    assert (
        AccuracyResult(num_correct_predictions=7, num_predictions=10).num_correct_predictions == 7
    )


def test_accuracy_result_num_predictions() -> None:
    assert AccuracyResult(num_correct_predictions=7, num_predictions=10).num_predictions == 10


def test_accuracy_result_is_frozen() -> None:
    result = AccuracyResult(num_correct_predictions=7, num_predictions=10)
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'num_predictions'"):
        result.num_predictions = 5


def test_accuracy_result_invalid_num_predictions_negative() -> None:
    with pytest.raises(ValueError, match="num_predictions"):
        AccuracyResult(num_correct_predictions=0, num_predictions=-1)


def test_accuracy_result_invalid_num_correct_predictions_negative() -> None:
    with pytest.raises(ValueError, match="num_correct_predictions"):
        AccuracyResult(num_correct_predictions=-1, num_predictions=10)


def test_accuracy_result_invalid_num_correct_predictions_exceeds_num_predictions() -> None:
    with pytest.raises(ValueError, match="num_correct_predictions"):
        AccuracyResult(num_correct_predictions=11, num_predictions=10)


def test_accuracy_result_repr() -> None:
    assert (
        repr(AccuracyResult(num_correct_predictions=7, num_predictions=10))
        == "AccuracyResult(num_correct_predictions=7, num_predictions=10)"
    )


def test_accuracy_result_str() -> None:
    assert (
        repr(AccuracyResult(num_correct_predictions=7, num_predictions=10))
        == "AccuracyResult(num_correct_predictions=7, num_predictions=10)"
    )


def test_accuracy_result_accuracy() -> None:
    assert AccuracyResult(num_correct_predictions=7, num_predictions=10).accuracy == 0.7


def test_accuracy_result_accuracy_zero() -> None:
    assert AccuracyResult(num_correct_predictions=0, num_predictions=10).accuracy == 0.0


def test_accuracy_result_accuracy_perfect() -> None:
    assert AccuracyResult(num_correct_predictions=10, num_predictions=10).accuracy == 1.0


def test_accuracy_result_accuracy_no_predictions() -> None:
    assert math.isnan(AccuracyResult(num_correct_predictions=0, num_predictions=0).accuracy)


def test_accuracy_result_accuracy_num_correct_predictions_is_nan() -> None:
    assert math.isnan(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=10).accuracy
    )


def test_accuracy_result_combine() -> None:
    result = AccuracyResult(num_correct_predictions=7, num_predictions=10).combine(
        AccuracyResult(num_correct_predictions=3, num_predictions=5)
    )
    assert result.equal(AccuracyResult(num_correct_predictions=10, num_predictions=15))


def test_accuracy_result_combine_empty() -> None:
    result = AccuracyResult(num_correct_predictions=7, num_predictions=10).combine(
        AccuracyResult(num_correct_predictions=0, num_predictions=0)
    )
    assert result.equal(AccuracyResult(num_correct_predictions=7, num_predictions=10))


def test_accuracy_result_combine_nan() -> None:
    result = AccuracyResult(num_correct_predictions=7, num_predictions=10).combine(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=5)
    )
    assert result.equal(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=15), equal_nan=True
    )


def test_accuracy_result_combine_returns_new_object() -> None:
    original = AccuracyResult(num_correct_predictions=7, num_predictions=10)
    combined = original.combine(AccuracyResult(num_correct_predictions=3, num_predictions=5))
    assert combined is not original
    assert original.equal(AccuracyResult(num_correct_predictions=7, num_predictions=10))
    assert combined.equal(AccuracyResult(num_correct_predictions=10, num_predictions=15))


def test_accuracy_result_combine_incorrect_object() -> None:
    result = AccuracyResult(num_correct_predictions=7, num_predictions=10)
    with pytest.raises(TypeError, match="Cannot combine AccuracyResult with"):
        result.combine({"num_correct_predictions": 0, "num_predictions": 0})


def test_accuracy_result_allclose_true() -> None:
    assert AccuracyResult(num_correct_predictions=7, num_predictions=10).allclose(
        AccuracyResult(num_correct_predictions=7, num_predictions=10)
    )


def test_accuracy_result_allclose_false_different_num_correct_predictions() -> None:
    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).allclose(
        AccuracyResult(num_correct_predictions=6, num_predictions=10)
    )


def test_accuracy_result_allclose_false_different_num_predictions() -> None:
    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).allclose(
        AccuracyResult(num_correct_predictions=7, num_predictions=11)
    )


def test_accuracy_result_allclose_false_different_type() -> None:
    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).allclose(
        {"num_correct_predictions": 7, "num_predictions": 10}
    )


def test_accuracy_result_allclose_false_different_type_child() -> None:
    class Child(AccuracyResult): ...

    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).allclose(
        Child(num_correct_predictions=7, num_predictions=10)
    )


def test_accuracy_result_allclose_atol() -> None:
    assert AccuracyResult(num_correct_predictions=7, num_predictions=10).allclose(
        AccuracyResult(num_correct_predictions=7, num_predictions=10), atol=1e-3
    )


def test_accuracy_result_allclose_rtol() -> None:
    assert AccuracyResult(num_correct_predictions=7, num_predictions=10).allclose(
        AccuracyResult(num_correct_predictions=7, num_predictions=10), rtol=1e-3
    )


def test_accuracy_result_equal_true() -> None:
    assert AccuracyResult(num_correct_predictions=7, num_predictions=10).equal(
        AccuracyResult(num_correct_predictions=7, num_predictions=10)
    )


def test_accuracy_result_equal_false_different_num_correct_predictions() -> None:
    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).equal(
        AccuracyResult(num_correct_predictions=6, num_predictions=10)
    )


def test_accuracy_result_equal_false_different_num_predictions() -> None:
    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).equal(
        AccuracyResult(num_correct_predictions=7, num_predictions=11)
    )


def test_accuracy_result_equal_false_different_type() -> None:
    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).equal(
        {"num_correct_predictions": 7, "num_predictions": 10}
    )


def test_accuracy_result_equal_false_different_type_child() -> None:
    class Child(AccuracyResult): ...

    assert not AccuracyResult(num_correct_predictions=7, num_predictions=10).equal(
        Child(num_correct_predictions=7, num_predictions=10)
    )


@pytest.mark.parametrize(
    ("prefix", "suffix", "expected"),
    [
        pytest.param(
            "",
            "",
            {"accuracy": 0.7, "num_correct_predictions": 7, "num_predictions": 10},
            id="no_prefix_suffix",
        ),
        pytest.param(
            "train_",
            "",
            {
                "train_accuracy": 0.7,
                "train_num_correct_predictions": 7,
                "train_num_predictions": 10,
            },
            id="prefix",
        ),
        pytest.param(
            "",
            "_train",
            {
                "accuracy_train": 0.7,
                "num_correct_predictions_train": 7,
                "num_predictions_train": 10,
            },
            id="suffix",
        ),
        pytest.param(
            "train_",
            "_v1",
            {
                "train_accuracy_v1": 0.7,
                "train_num_correct_predictions_v1": 7,
                "train_num_predictions_v1": 10,
            },
            id="prefix_and_suffix",
        ),
    ],
)
def test_accuracy_result_to_dict(prefix: str, suffix: str, expected: dict[str, float]) -> None:
    assert (
        AccuracyResult(num_correct_predictions=7, num_predictions=10).to_dict(
            prefix=prefix, suffix=suffix
        )
        == expected
    )


def test_accuracy_result_to_dict_no_predictions() -> None:
    assert objects_are_equal(
        AccuracyResult(num_correct_predictions=0, num_predictions=0).to_dict(),
        {"accuracy": float("nan"), "num_correct_predictions": 0, "num_predictions": 0},
        equal_nan=True,
    )


def test_accuracy_result_to_dict_nan() -> None:
    assert objects_are_equal(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=10).to_dict(),
        {"accuracy": float("nan"), "num_correct_predictions": float("nan"), "num_predictions": 10},
        equal_nan=True,
    )


def test_accuracy_result_to_str() -> None:
    assert AccuracyResult(num_correct_predictions=7, num_predictions=10).to_str() == (
        "[██████████████░░░░░░]  0.7000  (7/10)"
    )


def test_accuracy_result_to_str_perfect() -> None:
    assert AccuracyResult(num_correct_predictions=10, num_predictions=10).to_str() == (
        "[████████████████████]  1.0000  (10/10)"
    )


def test_accuracy_result_to_str_zero() -> None:
    assert AccuracyResult(num_correct_predictions=0, num_predictions=10).to_str() == (
        "[░░░░░░░░░░░░░░░░░░░░]  0.0000  (0/10)"
    )


def test_accuracy_result_to_str_large_numbers() -> None:
    assert AccuracyResult(num_correct_predictions=1000, num_predictions=10000).to_str() == (
        "[██░░░░░░░░░░░░░░░░░░]  0.1000  (1,000/10,000)"
    )


def test_accuracy_result_to_str_empty() -> None:
    assert AccuracyResult(num_correct_predictions=0, num_predictions=0).to_str() == (
        "AccuracyResult: no predictions"
    )


def test_accuracy_result_to_str_nan() -> None:
    assert AccuracyResult(num_correct_predictions=float("nan"), num_predictions=10).to_str() == (
        "AccuracyResult: unknown number of correct predictions"
    )
