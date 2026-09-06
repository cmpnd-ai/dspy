import pytest

from dspy.evaluate import normalize_text
from dspy.predict.aggregation import majority
from dspy.primitives.prediction import Completions, Prediction


def test_majority_with_prediction():
    prediction = Prediction.from_completions([{"answer": "2"}, {"answer": "2"}, {"answer": "3"}])
    result = majority(prediction)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_completions():
    completions = Completions([{"answer": "2"}, {"answer": "2"}, {"answer": "3"}])
    result = majority(completions)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_list():
    completions = [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
    result = majority(completions)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_normalize():
    completions = [{"answer": "2"}, {"answer": " 2"}, {"answer": "3"}]
    result = majority(completions, normalize=normalize_text)
    assert result.completions[0]["answer"] == "2"


def test_majority_with_field():
    completions = [
        {"answer": "2", "other": "1"},
        {"answer": "2", "other": "1"},
        {"answer": "3", "other": "2"},
    ]
    result = majority(completions, field="other")
    assert result.completions[0]["other"] == "1"


def test_majority_with_no_majority():
    completions = [{"answer": "2"}, {"answer": "3"}, {"answer": "4"}]
    result = majority(completions)
    assert result.completions[0]["answer"] == "2"  # The first completion is returned in case of a tie


def test_majority_all_ignored_should_not_return_ignored_completion():
    # Per the docstring, "When normalize returns None, that completion is ignored."
    # With default_normalize, empty/whitespace answers all normalize to None, so every
    # completion should be ignored and there is no majority to elect.
    completions = [{"answer": ""}, {"answer": " "}, {"answer": "\t"}]
    try:
        result = majority(completions)
    except (ValueError, IndexError):
        return  # acceptable: signal "no majority" rather than return an ignored completion
    # If a Prediction is returned, it must NOT be a completion that the contract says to ignore.
    assert result.completions[0]["answer"] != ""


def test_majority_all_ignored_raises_valueerror():
    # The all-ignored case must be signalled explicitly rather than silently returning
    # an ignored completion. The fix raises ValueError to signal "no majority".
    completions = [{"answer": ""}, {"answer": "   "}, {"answer": "\n"}]
    with pytest.raises(ValueError, match="every completion was ignored"):
        majority(completions)


def test_majority_all_ignored_with_prediction_input():
    # The Prediction input path (used by reduce_fn(outputs) in EnsembledProgram.forward)
    # must also signal "no majority" instead of returning an ignored completion.
    prediction = Prediction.from_completions([{"answer": ""}, {"answer": "  "}, {"answer": "\t"}])
    with pytest.raises(ValueError, match="every completion was ignored"):
        majority(prediction)


def test_majority_partial_ignore_still_counts_non_ignored():
    # When some completions are ignored (normalize to None) but others survive, only the
    # surviving (non-ignored) completions should be counted toward the majority.
    completions = [{"answer": ""}, {"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
    result = majority(completions)
    # The empty answer is ignored; "2" is the majority among the non-ignored completions.
    assert result.completions[0]["answer"] == "2"


def test_majority_partial_ignore_ignores_empty_for_tie():
    # An ignored completion should not be eligible as a tie-breaker either. Here the
    # non-ignored completions tie 1-1, so the earlier non-ignored completion ("3") wins,
    # and the ignored empty completion is never returned.
    completions = [{"answer": ""}, {"answer": "3"}, {"answer": "2"}]
    result = majority(completions)
    assert result.completions[0]["answer"] == "3"
