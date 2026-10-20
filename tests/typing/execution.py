"""Consumer contracts checked by ty, not executed by pytest.

Expected diagnostics on invalid examples also fail if an API regresses to Any/Unknown.
"""

from collections.abc import Iterator
from typing import Literal

from typing_extensions import assert_type

import dspy
from dspy.evaluate.metric_context import accepts_metric_context, call_metric
from dspy.teleprompt.bootstrap_trace import FailedPrediction, TraceData, bootstrap_trace_data
from dspy.utils.callback import BaseCallback
from dspy.utils.callback_context import ACTIVE_INVOCATION
from dspy.utils.execution import ExecutionRecord, ExecutionSpan, InvocationKind, JSONValue, UsageTotals


def score(gold: dspy.Example, pred: dspy.Prediction, *, metric_context: dspy.MetricContext) -> float:
    return 0.5


def feedback(gold: dspy.Example, pred: dspy.Prediction, *, metric_context: dspy.MetricContext) -> dspy.Prediction:
    return dspy.Prediction(score=0.5, feedback="Check the cited source")


def accept(gold: dspy.Example, pred: dspy.Prediction, trace: object = None) -> bool:
    return True


def check_consumers(program: dspy.Module, gold: dspy.Example, pred: dspy.Prediction) -> None:
    with dspy.capture_execution(program) as capture:
        program(**gold.inputs())
    assert_type(capture.roots, tuple[dspy.Invocation, ...])
    assert_type(capture.invocation, dspy.Invocation)
    assert_type(capture.trace, dspy.ExecutionTrace)
    root = capture.roots[0]
    assert_type(root.kind, InvocationKind)
    assert_type(root.descendants(), Iterator[dspy.Invocation])
    assert_type(root.children, tuple[dspy.Invocation, ...] | None)
    assert_type(root.trace, tuple[dspy.Invocation, ...] | None)
    assert_type(root.inputs, dict[str, object] | None)
    assert_type(root.outputs, object)
    assert_type(root.instance, object)
    assert_type(root.usage, UsageTotals | None)
    assert_type(root.own_usage, UsageTotals)
    assert_type(root.get_total_cost(), float | None)
    assert_type(root.get_total_lm_calls(), int)
    if root.trace is not None:
        assert_type(root.trace[0], dspy.Invocation)
    if root.usage is not None:
        assert_type(root.usage["model"]["total_tokens"], object)

    context = dspy.MetricContext(purpose="feedback", invocation=root, target=root)
    assert_type(context.purpose, Literal["score", "feedback"])
    assert_type(context.target, dspy.Invocation | None)
    assert_type(call_metric(score, gold, pred, context), float)
    assert_type(call_metric(feedback, gold, pred, context), dspy.Prediction)
    assert_type(call_metric(accept, gold, pred, context, legacy_args=(None,)), bool)
    assert_type(accepts_metric_context(score), bool)
    assert_type(ACTIVE_INVOCATION.get(), dspy.Invocation | None)

    record = capture.trace.to_dict()
    assert_type(record, ExecutionRecord)
    assert_type(record["trace_id"], str)
    assert_type(record["closed"], bool)
    assert_type(capture.trace.spans, tuple[ExecutionSpan, ...])
    span = capture.trace.span(root.call_id)
    assert_type(span["kind"], InvocationKind)
    assert_type(span["outputs"], JSONValue)
    if "cost" in span:
        assert_type(span["cost"], float | None)
    assert_type(capture.trace.ref(root.call_id), dspy.ExecutionRef)
    assert_type(capture.trace.annotate("Check this", span_id=root.call_id), dspy.Feedback)
    assert_type(dspy.ExecutionTrace.from_dict(record), dspy.ExecutionTrace)

    rows = bootstrap_trace_data(program, [gold], metric=score, capture_execution=True)
    assert_type(rows, list[TraceData])
    assert_type(rows[0]["invocation"], dspy.Invocation)
    assert_type(rows[0]["prediction"], dspy.Prediction | FailedPrediction)
    if "execution" in rows[0]:
        assert_type(rows[0]["execution"], dspy.ExecutionTrace)

    dspy.MetricContext(purpose="accept")  # ty: ignore[invalid-argument-type]
    dspy.MetricContext(target=0)  # ty: ignore[invalid-argument-type]
    dspy.MetricContext(invocation=capture.trace)  # ty: ignore[invalid-argument-type]
    context.invocation.get_total_cost()  # ty: ignore[unresolved-attribute]
    root.get_total_cost() + 1.0  # ty: ignore[unsupported-operator]
    _ = root.outputs.answer  # ty: ignore[unresolved-attribute]
    root.kind = "llm"  # ty: ignore[invalid-assignment]
    call_metric(score, gold, pred, root)  # ty: ignore[invalid-argument-type]
    BaseCallback().on_end("not an invocation")  # ty: ignore[invalid-argument-type]
    ACTIVE_INVOCATION.set("not an invocation")  # ty: ignore[invalid-argument-type]
    dspy.capture_execution(123)  # ty: ignore[invalid-argument-type]
    capture.trace.span(123)  # ty: ignore[invalid-argument-type]
    span["parent_span_id"] = 123  # ty: ignore[invalid-assignment]
    span["kind"] = "llm"  # ty: ignore[invalid-assignment]
