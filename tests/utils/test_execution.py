"""Execution evidence through real DSPy call, worker, and optimizer boundaries."""

import asyncio
import json
import threading
from dataclasses import asdict

import pytest

import dspy
from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


def test_capture_shares_callback_ids_but_not_callback_configuration_or_payload_references():
    observed = []

    class Observer(BaseCallback):
        def on_module_start(self, call_id, instance, inputs):
            observed.append(call_id)

    class Program(dspy.Module):
        def forward(self, values):
            with dspy.context(callbacks=[]):
                dspy.Tool(lambda x: x + 1, name="increment")(x=3)
            return dspy.Prediction(values=values)

    values = ["before"]
    with dspy.context(callbacks=[Observer()]), dspy.capture_execution() as capture:
        result = Program()(values=values)
    result["values"].append("after")
    record = capture.trace
    root, tool = record.spans
    assert observed == [root["span_id"]]
    assert tool["parent_span_id"] == root["span_id"]
    assert tool["outputs"] == 4
    assert root["outputs"] == {"values": ["before"]}
    detached = record.to_dict()
    detached["spans"][0]["outputs"]["values"].append("external mutation")
    assert record.span(root["span_id"])["outputs"] == {"values": ["before"]}
    assert dspy.ExecutionTrace.from_dict(json.loads(json.dumps(record.to_dict()))) == record


def test_failure_is_recorded_without_changing_the_exception():
    class Stop(BaseException):
        pass

    class Failing(dspy.Module):
        def forward(self):
            raise Stop("failure before prediction")

    with pytest.raises(Stop, match="failure before prediction"), dspy.capture_execution() as capture:
        Failing()()
    span, = capture.trace.spans
    assert span["error"] == {"type": "Stop", "message": "failure before prediction"}
    assert span["outputs"] is None
    assert span["end_time"] >= span["start_time"]


@pytest.mark.asyncio
async def test_concurrent_captures_and_joined_workers_have_correct_parents():
    class Child(dspy.Module):
        def forward(self, q):
            return dspy.Prediction(a=q)

    class Parent(dspy.Module):
        def forward(self, q):
            return dspy.Parallel(num_threads=2)([(Child(), {"q": q}), (Child(), {"q": q + "!"})])

    async def run(q):
        with dspy.capture_execution() as capture:
            await dspy.asyncify(Parent())(q=q)
        return capture.trace

    records = await asyncio.gather(run("alpha"), run("beta"))
    assert records[0].trace_id != records[1].trace_id
    for record, q in zip(records, ("alpha", "beta"), strict=False):
        spans = {s["span_id"]: s for s in record.spans}
        children = [s for s in spans.values() if s["name"].endswith("Child")]
        assert sorted(s["outputs"]["a"] for s in children) == [q, q + "!"]
        for child in children:
            # Parallel is an uninstrumented executor, not an extra module invocation.
            assert spans[child["parent_span_id"]]["name"].endswith("Parent")
        assert all(s["end_time"] is not None for s in spans.values())


def test_all_attempts_are_observed_but_only_winner_is_a_demonstration():
    class Candidate(dspy.Module):
        def __init__(self):
            self.answer = dspy.Predict("q -> a")

        def forward(self, q):
            return self.answer(q=q)

    scores = iter([0.8, 0.2])
    candidate = dspy.BestOfN(Candidate(), N=2, reward_fn=lambda inputs, pred: next(scores), threshold=1)
    with dspy.context(lm=DummyLM([{"a": "same"}, {"a": "same"}]), trace=[]), dspy.capture_execution(candidate) as capture:
        candidate(q="repeat")
        chosen, = dspy.settings.trace
    record = capture.trace
    calls = [s for s in record.spans if s["component_name"] == "module.answer"]
    assert len(calls) == 2
    assert all(s["error"] is None and s["outputs"] == {"a": "same"} for s in calls)
    assert calls[0]["span_id"] != calls[1]["span_id"]
    context = dspy.MetricContext("feedback", trace=[chosen], target=0, execution=record)
    assert context.target_ref.span_id == calls[0]["span_id"]


def test_candidate_records_survive_for_optimizer_research_but_exclude_outer_judge():
    class Candidate(dspy.Module):
        def forward(self, q):
            value = dspy.Tool(lambda text: text.upper(), name="uppercase")(text=q)
            return dspy.Prediction(a=value)

    def metric(gold, pred, *, metric_context):
        assert metric_context.purpose == "score"
        assert metric_context.trace == ()  # Deterministic work is not absence of execution.
        assert len(metric_context.execution.spans) == 2
        dspy.Tool(lambda: "judge work", name="judge")()
        return dspy.Prediction(score=0.25, feedback="retain metadata")

    program = Candidate()
    forward = program.forward
    data = bootstrap_trace_data(
        program, [dspy.Example(q="hello").with_inputs("q")], metric=metric, num_threads=1,
    )
    assert program.forward == forward
    assert data[0]["score"].feedback == "retain metadata"
    assert data[0]["execution"].to_dict()["closed"]
    assert [s["outputs"] for s in data[0]["execution"].spans] == [{"a": "HELLO"}, "HELLO"]


def test_delayed_feedback_resolves_exact_output_after_serialization():
    with dspy.capture_execution() as capture:
        dspy.Tool(lambda: {"a/b": [{"~key": "first"}, {"~key": "second"}]}, name="report")()
    record = dspy.ExecutionTrace.from_dict(json.loads(json.dumps(capture.trace.to_dict())))
    span, = record.spans
    feedback = record.annotate("Unsupported second claim", span_id=span["span_id"], output_path="/a~1b/1/~0key")
    wire = json.loads(json.dumps(asdict(feedback)))
    restored = dspy.Feedback(dspy.ExecutionRef(**wire["target"]), wire["content"])
    assert record.resolve(restored.target) == "second"
    assert record.resolve(record.ref(span["span_id"], output_path="")) == span["outputs"]
    assert record.resolve(record.ref()) == record.to_dict()
    with pytest.raises(ValueError, match="another execution"):
        record.resolve(dspy.ExecutionRef("foreign", span["span_id"]))
    with pytest.raises(KeyError):
        record.ref("missing")
    for path in ("/a~1b/01", "/a~1b/-1", "/a~1b/2", "/missing"):
        with pytest.raises(KeyError):
            record.ref(span["span_id"], output_path=path)
    for path in ("not/a/pointer", "/bad~2escape"):
        with pytest.raises(ValueError, match="Invalid JSON Pointer"):
            record.ref(span["span_id"], output_path=path)
    with pytest.raises(ValueError, match="requires an invocation"):
        record.ref(output_path="/a~1b")


@pytest.mark.asyncio
async def test_late_worker_cannot_rewrite_a_sealed_record():
    started, release = threading.Event(), threading.Event()

    class Slow(dspy.Module):
        def forward(self):
            started.set()
            assert release.wait(5)
            return dspy.Prediction(answer="late")

    try:
        with dspy.capture_execution() as capture:
            task = asyncio.create_task(dspy.asyncify(Slow())())
            assert await asyncio.to_thread(started.wait, 3)
        sealed = capture.trace.to_dict()
    finally:
        release.set()
    assert (await task).answer == "late"
    assert capture.trace.to_dict() == sealed
    unfinished, = capture.trace.spans
    assert unfinished["end_time"] is None
    with pytest.raises(ValueError, match="unfinished"):
        capture.trace.ref(unfinished["span_id"], output_path="")


@pytest.mark.asyncio
async def test_async_and_nested_capture_boundaries_are_isolated():
    class AsyncProgram(dspy.Module):
        async def aforward(self):
            return await dspy.Tool(lambda: "answer", name="answer").acall()

    with dspy.capture_execution() as outer:
        dspy.Tool(lambda: "outer", name="outer")()
        with dspy.capture_execution() as inner:
            assert await AsyncProgram().acall() == "answer"
    assert len(outer.trace.spans) == 1
    root, tool = inner.trace.spans
    assert root["parent_span_id"] is None
    assert tool["parent_span_id"] == root["span_id"]
    assert tool["outputs"] == "answer"


def test_parse_failure_is_attributed_to_the_failing_sibling():
    from dspy.utils.exceptions import AdapterParseError

    class Adapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            if inputs["q"] == "second":
                raise AdapterParseError(adapter_name="test", signature=signature, lm_response="bad", parsed_result={})
            return [{"a": "ok"}]

    class Program(dspy.Module):
        def __init__(self):
            self.first = dspy.Predict("q -> a")
            self.second = dspy.Predict("q -> a")

        def forward(self, q):
            self.first(q="first")
            return self.second(q="second")

    with dspy.context(lm=DummyLM([]), adapter=Adapter()):
        data = bootstrap_trace_data(
            Program(), [dspy.Example(q="root").with_inputs("q")], capture_execution=True, num_threads=1,
        )[0]
    context = dspy.MetricContext("feedback", trace=data["trace"], target=1, execution=data["execution"])
    failed = context.execution.resolve(context.target_ref)
    assert failed["component_name"] == "second"
    assert failed["error"]["type"] == "AdapterParseError"
    assert data["trace"][1][1] == {"q": "second"}
