import pytest

import dspy
from dspy.predict.best_of_n import BestOfN
from dspy.predict.predict import Predict
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

    best_of_n = BestOfN(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = best_of_n(question="What is the capital of Belgium?")

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

    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    # fail_count defaults to N: up to N failures are tolerated, so all N failing
    # rollouts are absorbed and the best prediction seen so far (None) is returned.
    result = best_of_n(question="What is the capital of Belgium?")
    assert result is None
    # the instance budget must not leak across forward() calls
    assert best_of_n.fail_count == 3


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

    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )


def test_best_of_n_fail_count_tolerates_single_late_failure():
    # threshold=2.0 is unreachable (reward capped at 1.0), so all N rollouts run and
    # the single failure on the final rollout is actually reached. With
    # fail_count=2 a single late failure must be tolerated (not re-raised).
    dspy.configure(lm=DummyLM([{"answer": "Brussels"}] * 4))

    rollout_count = [0]

    def succeed_four_then_fail(self, **kwargs):
        rollout_count[0] += 1
        if rollout_count[0] <= 4:
            return self.predictor(**kwargs)
        raise ValueError("transient failure on rollout %d" % rollout_count[0])

    predict = DummyModule("question -> answer", succeed_four_then_fail)
    best_of_n = BestOfN(module=predict, N=5, reward_fn=lambda _, __: 1.0, threshold=2.0, fail_count=2)

    result = best_of_n(question="What is the capital of Belgium?")

    assert result.answer == "Brussels"
    assert best_of_n.fail_count == 2


def test_best_of_n_fail_count_still_raises_when_exceeded():
    # A single late failure is tolerated with fail_count=1, but the *second*
    # late failure exceeds the budget and must be re-raised.
    dspy.configure(lm=DummyLM([{"answer": "Brussels"}] * 4))

    rollout_count = [0]

    def succeed_four_then_fail_twice(self, **kwargs):
        rollout_count[0] += 1
        if rollout_count[0] <= 4:
            return self.predictor(**kwargs)
        raise ValueError("failure on rollout %d" % rollout_count[0])

    predict = DummyModule("question -> answer", succeed_four_then_fail_twice)
    best_of_n = BestOfN(module=predict, N=6, reward_fn=lambda _, __: 1.0, threshold=2.0, fail_count=1)

    with pytest.raises(ValueError, match="failure on rollout 6"):
        best_of_n(question="What is the capital of Belgium?")
    assert best_of_n.fail_count == 1


def test_best_of_n_fail_count_does_not_leak_across_calls():
    # Each forward() call fails twice then succeeds; with fail_count=2 both
    # leading failures must be tolerated on *every* call (no cross-call leak).
    dspy.configure(lm=DummyLM([{"answer": "Brussels"}] * 20))

    state = {"rollout_in_call": 0}

    def fail_twice_per_call(self, **kwargs):
        if state["rollout_in_call"] < 2:
            state["rollout_in_call"] += 1
            raise ValueError("leading failure #%d" % state["rollout_in_call"])
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", fail_twice_per_call)
    best_of_n = BestOfN(module=predict, N=5, reward_fn=lambda _, __: 1.0, threshold=2.0, fail_count=2)

    r1 = best_of_n(question="Q?")
    assert r1.answer == "Brussels"
    assert best_of_n.fail_count == 2

    state["rollout_in_call"] = 0
    r2 = best_of_n(question="Q?")
    assert r2.answer == "Brussels"
    assert best_of_n.fail_count == 2
