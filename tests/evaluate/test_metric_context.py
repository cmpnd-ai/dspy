"""Public metric requests: purpose is independent of evidence and results stay caller-owned."""

import pytest

import dspy
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data
from dspy.utils.dummies import DummyLM


class Echo(dspy.Module):
    def forward(self, q):
        return dspy.Prediction(a=q)


def example():
    return dspy.Example(q="hello", a="hello").with_inputs("q")


@pytest.mark.parametrize("as_module", [False, True])
def test_evaluate_opt_in_preserves_result(as_module):
    verdict = dspy.Prediction(score=0.25, feedback="keep this", objective_scores={"cost": -2})

    def metric(gold, pred, *, metric_context):
        assert metric_context.purpose == "score"
        assert metric_context.trace is None
        return verdict

    class Judge(dspy.Module):
        def forward(self, gold, pred, *, metric_context):
            return metric(gold, pred, metric_context=metric_context)

    result = dspy.Evaluate(devset=[example()], metric=Judge() if as_module else metric, num_threads=1)(Echo())
    assert result.score == 25.0
    assert result.results[0][2] is verdict


def test_kwargs_is_not_opt_in_and_type_error_is_not_retried():
    calls = []

    def legacy(gold, pred, **kwargs):
        assert kwargs == {}
        return 0.25

    assert dspy.Evaluate(devset=[example()], metric=legacy, num_threads=1)(Echo()).score == 25.0

    def raising(gold, pred, *, metric_context):
        calls.append(metric_context.purpose)
        raise TypeError("inside metric")

    with pytest.raises(Exception, match="Execution cancelled due to errors"):
        dspy.Evaluate(devset=[example()], metric=raising, num_threads=1, max_errors=1)(Echo())
    assert calls == ["score"]


@pytest.mark.parametrize("threshold,purpose", [(None, "accept"), (0, "accept"), (0.5, "score")])
@pytest.mark.parametrize("predictor", [False, True])
def test_bootstrap_purpose_does_not_follow_evidence(threshold, purpose, predictor):
    seen = []

    def metric(gold, pred, *, metric_context):
        seen.append(metric_context)
        return False if metric_context.purpose == "accept" else 0.75

    with dspy.context(lm=DummyLM([{"a": "hello"}])):
        compiled = dspy.BootstrapFewShot(
            metric=metric, metric_threshold=threshold, max_bootstrapped_demos=1, max_labeled_demos=0,
        ).compile(dspy.Predict("q -> a") if predictor else Echo(), trainset=[example()])
    assert len(seen) == 1
    assert seen[0].purpose == purpose
    assert len(seen[0].trace) == int(predictor)
    if predictor:
        assert len(compiled.demos) == int(purpose == "score")


def test_captured_reward_and_finetune_acceptance_are_different_requests(monkeypatch):
    seen = []

    def metric(gold, pred, *, metric_context):
        seen.append(metric_context)
        return pred.a == "keep" if metric_context.purpose == "accept" else 0.25

    data = [dspy.Example(q=q, a=q).with_inputs("q") for q in ("keep", "drop")]
    lm = DummyLM({"keep": {"a": "keep"}, "drop": {"a": "drop"}})
    student = dspy.Predict("q -> a")
    student.set_lm(lm)
    # This is the trace collector used for GRPO rewards.
    rewards = bootstrap_trace_data(student, data, metric=metric, num_threads=1)
    assert [r["score"] for r in rewards] == [0.25, 0.25]
    assert [c.purpose for c in seen] == ["score", "score"]
    assert all(len(c.trace) == 1 for c in seen)
    seen.clear()

    class DataAdapter(dspy.ChatAdapter):
        def format_finetune_data(self, signature, demos, inputs, outputs):
            return {"q": inputs["q"], "a": outputs["a"]}

    training_data = []

    def train(jobs):
        for job in jobs.values():
            training_data.extend(job["train_data"])
        return {key: job["lm"] for key, job in jobs.items()}

    optimizer = dspy.BootstrapFinetune(metric=metric, adapter=DataAdapter(), num_threads=1)
    monkeypatch.setattr(optimizer, "finetune_lms", train)  # provider boundary only
    optimizer.compile(student, trainset=data)
    assert [c.purpose for c in seen] == ["accept", "accept"]
    assert all(len(c.trace) == 1 for c in seen)
    assert training_data == [{"q": "keep", "a": "keep"}]


@pytest.mark.parametrize("strategy", [dspy.BestOfN, dspy.Refine])
def test_reward_uses_inputs_and_excludes_judge(strategy):
    seen = []

    def reward(inputs, pred, *, metric_context):
        assert inputs == {"q": "hello"}
        seen.append(metric_context)
        dspy.Predict("q -> a")(q="judge")
        return 0.75

    with dspy.context(lm=DummyLM([{"a": "hello"}, {"a": "judge"}]), trace=[]):
        result = strategy(dspy.Predict("q -> a"), N=1, reward_fn=reward, threshold=1)(q="hello")
        assert result.a == "hello"
        assert [step[1] for step in dspy.settings.trace] == [{"q": "hello"}]
    assert seen[0].purpose == "score"
    assert [step[1] for step in seen[0].trace] == [{"q": "hello"}]


@pytest.mark.parametrize("seed,target", [(0, 2), (1, 0)])
def test_gepa_feedback_identifies_named_component_and_occurrence(monkeypatch, seed, target):
    import gepa

    class Program(dspy.Module):
        def __init__(self):
            self.first = dspy.Predict("q -> a")
            self.sibling = dspy.Predict("q -> a")

        def forward(self, q):
            self.first(q="first call")
            self.sibling(q="sibling call")
            return self.first(q="last call")

    feedback_results = []

    def metric(gold, pred, *, metric_context):
        if metric_context.purpose == "score":
            return dspy.Prediction(score=0.25, objective_scores={"cost": -2})
        assert metric_context.purpose == "feedback"
        assert metric_context.component_name == "first"
        assert metric_context.target == target
        assert len(metric_context.trace) == 3
        step = metric_context.trace[metric_context.target]
        assert step[1]["q"] == ("last call" if target == 2 else "first call")
        result = dspy.Prediction(score=0.9, feedback=f"Improve {step[1]['q']}")
        feedback_results.append(result)
        return result

    class Finished(Exception):
        pass

    def optimize(*, adapter, seed_candidate, trainset, **kwargs):
        batch = adapter.evaluate(trainset, seed_candidate, capture_traces=True)
        assert batch.scores == [0.25]
        assert batch.objective_scores == [{"cost": -2}]
        records = adapter.make_reflective_dataset(seed_candidate, batch, ["first"])
        assert records["first"][0]["Feedback"] == ("Improve last call" if target == 2 else "Improve first call")
        # Existing GEPA interpretation: component feedback cannot replace program fitness.
        assert feedback_results[0].score == 0.25
        raise Finished

    monkeypatch.setattr(gepa, "optimize", optimize)  # external optimizer boundary
    with dspy.context(lm=DummyLM([{"a": "first"}, {"a": "sibling"}, {"a": "last"}])):
        with pytest.raises(Finished):
            dspy.GEPA(metric=metric, max_metric_calls=4, reflection_lm=DummyLM([]), seed=seed).compile(
                Program(), trainset=[example()],
            )
