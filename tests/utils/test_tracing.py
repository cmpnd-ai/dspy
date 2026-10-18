import asyncio
from unittest.mock import patch

import pytest
from litellm import Choices, Message, ModelResponse

import dspy
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


def test_repeated_predictor_calls_have_shared_component_but_distinct_invocations_and_snapshots():
    class Program(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("items: list[str] -> answer: str")

        def forward(self, items):
            first = self.predict(items=items)
            items.append("second")
            second = self.predict(items=items)
            return first, second

    callback_ids = []

    class Callback(BaseCallback):
        def on_module_start(self, call_id, instance, inputs):
            callback_ids.append(call_id)

    program = Program()
    items = ["first"]
    with dspy.context(lm=DummyLM([{"answer": "one"}, {"answer": "two"}]), callbacks=[Callback()]):
        with dspy.capture_trace(program) as execution:
            first, _second = program(items=items)
    first.answer = "changed"
    items.append("after")

    root = execution.invocations[0]
    calls = execution.predictor_invocations()
    assert [i.call_id for i in [root, *calls]] == callback_ids
    assert len(set(callback_ids)) == 3
    assert all(i.parent_id == root.call_id for i in calls)
    assert all(i.component is program.predict for i in calls)
    assert [execution.component_name(i) for i in calls] == ["predict", "predict"]
    assert [step[1]["items"] for step in execution.predictor_trace()] == [["first"], ["first", "second"]]
    assert [step[2].answer for step in execution.predictor_trace()] == ["one", "two"]
    assert root.outputs[0].answer == "one"
    assert all(i.ended_at >= i.started_at for i in execution.invocations)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_parallel_calls_keep_parentage_without_callbacks(asynchronous):
    class Branch(dspy.Module):
        def forward(self, value):
            dspy.record_event("agent_state", {"value": value})
            return value * 3

        async def aforward(self, value):
            await asyncio.sleep(0)
            return self.forward(value)

    class Program(dspy.Module):
        def __init__(self):
            self.branch = Branch()

        def forward(self):
            return dspy.Parallel(num_threads=2, disable_progress_bar=True)(
                [(self.branch, {"value": 2}), (self.branch, {"value": 7})]
            )

        async def aforward(self):
            return await asyncio.gather(self.branch.acall(value=2), self.branch.acall(value=7))

    program = Program()
    with dspy.capture_trace(program) as execution:
        result = asyncio.run(program.acall()) if asynchronous else program()
    assert result == [6, 21]
    root, *children = execution.invocations
    assert len(children) == 2
    assert all(i.parent_id == root.call_id and i.run_id == root.run_id for i in children)
    assert sorted(i.data["agent_state"][0]["value"] for i in children) == [2, 7]


@pytest.mark.asyncio
async def test_cancellation_closes_invocations_and_restores_context():
    started = asyncio.Event()

    class Wait(dspy.Module):
        async def aforward(self):
            started.set()
            await asyncio.Event().wait()

    with dspy.capture_trace() as execution:
        task = asyncio.create_task(Wait().acall())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    (failed,) = execution.invocations
    assert failed.error["type"] == "CancelledError"
    assert failed.ended_at is not None
    with dspy.capture_trace() as other:
        pass
    assert other.parent_id is None
    assert other.run_id != execution.run_id


def test_best_of_n_retains_failed_and_discarded_attempts_but_selects_only_winner():
    class Program(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question):
            result = self.predict(question=question)
            if self.get_lm().kwargs["rollout_id"] == 1:
                raise ValueError("failed after LM work")
            result.rank = self.get_lm().kwargs["rollout_id"]
            return result

    program = dspy.BestOfN(Program(), N=3, reward_fn=lambda _, p: p.rank, threshold=10)
    with dspy.context(lm=DummyLM([{"answer": "done"}]), trace=None):
        with dspy.capture_trace(program) as execution:
            result = program(question="test")
    assert result.rank == 2
    assert len(execution.predictor_trace(selected=False)) == 3
    (selected,) = execution.predictor_invocations()
    assert selected.component.get_lm().kwargs["rollout_id"] == 2
    assert execution.component_name(selected) == "module.predict"
    assert any(i.error and i.error["message"] == "failed after LM work" for i in execution.invocations)


def test_usage_and_history_join_invocations_and_cache_hits_do_not_bill_twice():
    lm = dspy.LM("openai/test-model", engine="litellm", cache=True)
    response = ModelResponse(
        choices=[Choices(message=Message(content="answer"))],
        model="test-model",
        usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    )
    with patch("litellm.completion", return_value=response):
        with dspy.capture_trace() as execution:
            with dspy.capture_trace() as rejected:
                lm("same request")
            rejected.discard()
            lm("same request")
    usage = execution.usage()["openai/test-model"]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 3, 14)
    assert [i.data["lm_result"][0]["cache_hit"] for i in execution.invocations] == [False, True]
    assert [entry["call_id"] for entry in lm.history] == [i.call_id for i in execution.invocations]
    assert rejected.run_id == execution.run_id
    assert len(rejected.invocations) == 1


def test_evaluation_has_per_example_traces_and_explicit_metric_purpose():
    class Program(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question):
            if question == "bad":
                raise ValueError("bad input")
            return self.predict(question=question)

    def metric(example, prediction, *, metric_context):
        assert metric_context.purpose == "score"
        (step,) = metric_context.execution.predictor_trace()
        assert step[1]["question"] == example.question
        return prediction.answer == "yes"

    examples = [dspy.Example(question=q).with_inputs("question") for q in ["first", "bad", "third"]]
    with dspy.context(lm=DummyLM({"first": {"answer": "yes"}, "third": {"answer": "no"}})):
        result = dspy.Evaluate(devset=examples, metric=metric, num_threads=2, capture_execution=True)(Program())
    assert [score for _, _, score in result.results] == [True, 0.0, False]
    traces = [item.attempts[0].execution for item in result.executions]
    assert len({trace.run_id for trace in traces}) == 3
    assert traces[1].invocations[0].error["message"] == "bad input"
    assert result.executions[1].accepted is None


def test_bootstrap_capture_passes_execution_to_new_metrics_without_mutating_program():
    program = dspy.Predict("question -> answer")
    original_forward = program.forward
    contexts = []

    def metric(example, prediction, *, metric_context):
        contexts.append(metric_context)
        return 0.75

    example = dspy.Example(question="test").with_inputs("question")
    with dspy.context(lm=DummyLM([{"answer": "yes"}])):
        (data,) = bootstrap_trace_data(program, [example], metric=metric, num_threads=1)
    assert program.forward == original_forward
    assert contexts[0].purpose == "score"
    assert contexts[0].execution is data["execution"]
    assert data["trace"] == data["execution"].predictor_trace()
    assert data["score"] == 0.75


def test_bootstrap_metric_gets_explicit_acceptance_purpose():
    contexts = []

    def metric(example, prediction, *, metric_context):
        contexts.append(metric_context)
        return metric_context.purpose == "bootstrap" and prediction.answer == "yes"

    with dspy.context(lm=DummyLM([{"answer": "yes"}])):
        program = dspy.BootstrapFewShot(metric=metric, max_labeled_demos=0).compile(
            dspy.Predict("question -> answer"),
            trainset=[dspy.Example(question="test", answer="yes").with_inputs("question")],
        )
    assert len(program.demos) == 1
    assert contexts[0].purpose == "bootstrap"
    assert len(contexts[0].execution.predictor_trace()) == 1


def test_observers_are_detached_and_legacy_trace_keeps_reference_semantics():
    class Mutator(dspy.ExecutionObserver):
        def on_end(self, invocation):
            invocation.outputs = "corrupted"
            invocation.inputs.clear()
            raise RuntimeError("export failed")

    predictor = dspy.Predict("question -> answer")
    with dspy.context(lm=DummyLM([{"answer": "original"}]), trace=[], execution_observers=[Mutator()]):
        with dspy.capture_trace() as execution:
            result = predictor(question="test")
        assert dspy.settings.trace[0][2] is result
        result.answer = "later"
        assert dspy.settings.trace[0][2].answer == "later"
    assert execution.predictor_trace()[0][2].answer == "original"
    assert execution.invocations[0].inputs["kwargs"]["question"] == "test"


def test_legacy_evaluation_and_metric_typeerror_are_not_reinterpreted():
    from dspy.evaluate.metric_context import MetricContext, call_metric

    calls = []

    def broken(example, prediction, **kwargs):
        calls.append(kwargs)
        raise TypeError("inside the metric")

    with pytest.raises(TypeError, match="inside the metric"):
        call_metric(broken, None, None, MetricContext())
    assert calls == [{}]
    result = dspy.Evaluate(devset=[dspy.Example(x=1).with_inputs("x")], metric=lambda e, p: p.y, num_threads=1)(
        lambda x: dspy.Prediction(y=x)
    )
    assert set(result.keys()) == {"score", "results"}
    assert result.score == 100


def test_selection_is_independent_and_portable_export_omits_payloads():
    import json

    predictor = dspy.Predict("question -> answer")
    with dspy.context(lm=DummyLM([{"answer": "one"}, {"answer": "two"}])):
        with dspy.capture_trace() as execution:
            with dspy.capture_trace() as rejected:
                predictor(question="private")
            rejected.discard()
            predictor(question="public")
    first, second = execution.predictor_invocations(selected=False)
    selection = dspy.ExecutionSelection(frozenset([first.ref]))
    assert execution.predictor_trace(selection=selection)[0][2].answer == "one"
    assert execution.predictor_trace()[0][2].answer == "two"
    assert execution.get(second.ref) is second
    assert execution.complete
    assert "private" not in json.dumps(execution.to_dict())
    assert "private" in json.dumps(execution.to_dict(include_payloads=True))


@pytest.mark.asyncio
async def test_closed_capture_reports_outstanding_calls_until_they_finish():
    started, finish = asyncio.Event(), asyncio.Event()

    class Wait(dspy.Module):
        async def aforward(self):
            started.set()
            await finish.wait()
            return "done"

    with dspy.capture_trace() as execution:
        task = asyncio.create_task(Wait().acall())
        await started.wait()
    assert execution.closed and not execution.complete
    finish.set()
    assert await task == "done"
    assert execution.complete


def test_failed_metric_keeps_program_and_metric_provenance():
    def metric(e, p, *, metric_context):
        raise ValueError("judge failed")

    with dspy.context(lm=DummyLM([{"answer": "ok"}])):
        result = dspy.Evaluate(
            devset=[dspy.Example(question="x").with_inputs("question")], metric=metric, num_threads=1
        )(dspy.Predict("question -> answer"))
    item = result.executions[0]
    assert item.accepted is None
    assert item.attempts[0].program and item.attempts[0].metric
    assert item.attempts[0].program != item.attempts[0].metric
    assert item.attempts[0].execution is None
