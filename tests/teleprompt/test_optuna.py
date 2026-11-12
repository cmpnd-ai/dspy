import importlib.util

import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShotWithOptuna
from dspy.utils.dummies import DummyLM

if importlib.util.find_spec("optuna") is None:
    pytest.skip(reason="optuna is not installed", allow_module_level=True)

pytestmark = pytest.mark.extra


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


examples = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
]
trainset = [examples[0]]
valset = [examples[1]]


def _make_lm():
    return DummyLM([{"output": "blue"}] * 20)


def test_compile_without_max_demos():
    """The documented cheatsheet call (no max_demos) should succeed."""
    dspy.configure(lm=_make_lm())
    student = SimpleModule("input -> output")
    optimizer = BootstrapFewShotWithOptuna(
        metric=simple_metric,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        num_candidate_programs=2,
    )
    compiled = optimizer.compile(student=student, trainset=trainset, valset=valset)
    assert compiled is not None


def test_compile_with_explicit_max_demos():
    """Explicitly passing max_demos should still work and forward it to BootstrapFewShot."""
    dspy.configure(lm=_make_lm())
    student = SimpleModule("input -> output")
    optimizer = BootstrapFewShotWithOptuna(
        metric=simple_metric,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        num_candidate_programs=2,
    )
    compiled = optimizer.compile(student=student, trainset=trainset, valset=valset, max_demos=1)
    assert compiled is not None


def test_max_demos_defaults_to_init_max_bootstrapped_demos():
    """When max_demos is omitted, it should default to __init__'s max_bootstrapped_demos."""
    dspy.configure(lm=_make_lm())
    student = SimpleModule("input -> output")
    optimizer = BootstrapFewShotWithOptuna(
        metric=simple_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=1,
        num_candidate_programs=1,
    )

    import dspy.teleprompt.teleprompt_optuna as optuna_mod

    captured = {}
    original_bootstrap = optuna_mod.BootstrapFewShot

    class CapturingBootstrap(original_bootstrap):
        def __init__(self, **kwargs):
            captured["max_bootstrapped_demos"] = kwargs.get("max_bootstrapped_demos")
            super().__init__(**kwargs)

    optuna_mod.BootstrapFewShot = CapturingBootstrap
    try:
        optimizer.compile(student=student, trainset=trainset, valset=valset)
    finally:
        optuna_mod.BootstrapFewShot = original_bootstrap

    assert captured["max_bootstrapped_demos"] == 2
