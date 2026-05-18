from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from mlev.functional.array import accuracy
from mlev.results import AccuracyResult
from mlev.testing.fixtures import sklearn_available
from mlev.utils.imports import is_sklearn_available

if is_sklearn_available():
    from sklearn.metrics import accuracy_score
else:
    accuracy_score = Mock(side_effect=ValueError)

##############################
#     Tests for accuracy     #
##############################

# ----------------------------------------------------
# Comparison with sklearn accuracy_score
# ----------------------------------------------------


@sklearn_available
@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            id="all-correct-int",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([0, 1, 1, 0, 0]),
            id="all-incorrect-int",
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1, 1, 0, 1]),
            id="partial-correct-int",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
            id="all-correct-float",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0, 1.0]),
            id="partial-correct-float",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["cat", "dog", "cat", "dog"]),
            id="all-correct-str",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["dog", "cat", "dog", "cat"]),
            id="all-incorrect-str",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["cat", "cat", "cat", "dog"]),
            id="partial-correct-str",
        ),
        pytest.param(
            np.array(["cat", "dog", "bird", "fish", "cat"]),
            np.array(["cat", "dog", "bird", "bird", "dog"]),
            id="multiclass-str-partial",
        ),
        pytest.param(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 2, 1, 0, 0, 1]),
            id="multiclass-int-partial",
        ),
        pytest.param(
            np.array([1]),
            np.array([1]),
            id="single-correct",
        ),
        pytest.param(
            np.array([1]),
            np.array([0]),
            id="single-incorrect",
        ),
        pytest.param(
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
            id="all-same-correct",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 0]),
            np.array([1, 1, 1, 1, 1]),
            id="all-same-incorrect",
        ),
        pytest.param(
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            id="long-all-correct",
        ),
        pytest.param(
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            id="long-all-incorrect",
        ),
        pytest.param(
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1]),
            id="long-partial",
        ),
    ],
)
def test_accuracy_matches_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    result = accuracy(y_true=y_true, y_pred=y_pred)
    sklearn_num_correct = int(accuracy_score(y_true, y_pred, normalize=False))
    assert result.equal(
        AccuracyResult(
            num_correct_predictions=sklearn_num_correct,
            num_predictions=len(y_true),
        )
    )


@sklearn_available
@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([1.0, 0.0, np.nan, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            id="nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, np.nan, 0.0, 1.0]),
            id="nan-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, np.nan, 0.0, 1.0, 1.0, np.nan]),
            np.array([1.0, np.nan, 0.0, 1.0, 0.0, np.nan]),
            id="nan-in-both",
        ),
    ],
)
def test_accuracy_omit_matches_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # sklearn drops NaN rows implicitly via boolean mask
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    sklearn_num_correct = int(accuracy_score(y_true[mask], y_pred[mask], normalize=False))
    result = accuracy(y_true=y_true, y_pred=y_pred, missing_policy="omit")
    assert result.equal(
        AccuracyResult(
            num_correct_predictions=sklearn_num_correct,
            num_predictions=int(mask.sum()),
        )
    )
