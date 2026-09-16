import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.utils.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


def test_basic_workflow():
    """Test to ensure the basic compile flow runs without errors."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(
        [
            "Initial thoughts",
            "Finish[blue]",  # Expected output for both training and validation
        ]
    )
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]
    optimizer.compile(student, teacher=teacher, trainset=trainset)


def test_restrict_matching_no_candidate_seed_raises_clear_error():
    """restrict that matches no candidate seed should raise ValueError, not UnboundLocalError."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    ]

    with pytest.raises(ValueError, match="restrict"):
        optimizer.compile(student, teacher=teacher, trainset=trainset, restrict=[999])


def test_restrict_as_single_use_iterator_still_matches_a_valid_seed():
    """A single-use iterable `restrict` must not be exhausted by the upfront validation check."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    ]

    # -3 (zero-shot) is a valid seed; a plain iterator is single-use.
    result = optimizer.compile(student, teacher=teacher, trainset=trainset, restrict=iter([-3]))
    assert result is not None


def test_bootstrap_candidates_keep_seeded_shuffle_and_size_contracts():
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")
    trainset = list(range(6))
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=simple_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=2,
        num_candidate_programs=2,
        max_errors=7,
        metric_threshold=0.5,
    )

    programs = [student.reset_copy() for _ in range(3)]
    with (
        patch("dspy.teleprompt.random_search.BootstrapFewShot") as bootstrap,
        patch("dspy.teleprompt.random_search.Evaluate") as evaluate_cls,
    ):
        bootstrap.return_value.compile.side_effect = programs
        evaluate_cls.return_value.side_effect = [
            SimpleNamespace(score=score, results=[]) for score in (1, 2, 3)
        ]
        result = optimizer.compile(student, teacher=teacher, trainset=trainset, restrict=[-1, 0, 1])

    calls = bootstrap.call_args_list
    shuffled = []
    for seed in (0, 1):
        candidate_trainset = list(trainset)
        random.Random(seed).shuffle(candidate_trainset)
        shuffled.append(candidate_trainset)
    assert [call.kwargs["trainset"] for call in bootstrap.return_value.compile.call_args_list] == [
        trainset,
        *shuffled,
    ]
    assert [call.kwargs["max_bootstrapped_demos"] for call in calls] == [
        4,
        random.Random(0).randint(1, 4),
        random.Random(1).randint(1, 4),
    ]
    assert all(call.kwargs["teacher"] is teacher for call in bootstrap.return_value.compile.call_args_list)
    assert all(call.kwargs["max_errors"] == 7 for call in calls)
    assert all(call.kwargs["metric_threshold"] == 0.5 for call in calls)
    assert result is programs[-1]
