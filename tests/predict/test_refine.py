import pytest

import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


def test_refine_forward_success_first_attempt():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    reward_call_count = [0]

    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        # The answer should always be one word.
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)

    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = refine(question="What is the capital of Belgium?")

    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] > 0, "Reward function should have been called"
    assert module_call_count[0] == 3, (
        "Module should have been called exactly 3 times, but was called %d times" % module_call_count[0]
    )


def test_refine_module_default_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")


def test_refine_module_custom_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def raise_on_second_call(self, **kwargs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", raise_on_second_call)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )


def _make_fail_leading_module(answers, state):
    """Build a DummyModule that fails the first ``state[0]`` attempts of the current
    ``Refine.forward`` call, then succeeds. The test mutates ``state[0]`` before each
    ``forward`` call to script the per-call failure pattern. ``state`` is captured by
    closure so it survives the per-attempt ``module.deepcopy()`` in ``Refine.forward``.
    """

    def fail_leading(self, **kwargs):
        if state[0] > 0:
            state[0] -= 1
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    lm = DummyLM([{"answer": a} for a in answers])
    dspy.configure(lm=lm)
    return DummyModule("question -> answer", fail_leading)


def test_fail_count_attribute_not_mutated_by_forward():
    state = [0]
    predict = _make_fail_leading_module([f"ans{i}" for i in range(4)], state)
    refine = Refine(module=predict, N=4, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=2)

    # Tolerate one failure, then succeed. Under the bug, this decremented self.fail_count to 1.
    state[0] = 1
    assert refine(question="q?").answer == "ans0"
    assert refine.fail_count == 2, "fail_count must not be depleted across forward calls"


def test_fail_count_budget_resets_each_forward_call():
    state = [0]
    predict = _make_fail_leading_module([f"ans{i}" for i in range(4)], state)
    refine = Refine(module=predict, N=4, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=2)

    # Call 1 tolerates 1 failure then succeeds; burns 1 from the budget under the buggy version.
    state[0] = 1
    assert refine(question="q?").answer == "ans0"

    # Call 2 tolerates 2 failures then succeeds. Under the bug self.fail_count==1 here, so the
    # 2nd failure raises; under the fix the full per-call budget is available and this succeeds.
    state[0] = 2
    assert refine(question="q?").answer == "ans1"

    # Call 3 must still have the full budget available.
    state[0] = 2
    assert refine(question="q?").answer == "ans2"

    assert refine.fail_count == 2


def test_fail_count_still_exhausts_within_a_single_call():
    state = [0]
    predict = _make_fail_leading_module([f"ans{i}" for i in range(4)], state)
    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=1)

    # fail_count=1: the module must still raise once more than the budget of failures has occurred
    # within a single call (regression guard so the fix does not disable within-call exhaustion).
    state[0] = 3
    with pytest.raises(ValueError):
        refine(question="q?")
    assert refine.fail_count == 1
