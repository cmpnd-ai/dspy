import json

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import dspy
from dspy.utils.otel import OTelBridge
from dspy.utils.tracing import ExecutionObserver, capture_trace


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("payloads", [False, True])
def test_canonical_spans_and_context_restoration(fail, payloads):
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer(__name__)
    bridge = OTelBridge(processor, include_payloads=payloads)
    seen = []

    class Observer(ExecutionObserver):
        def on_end(self, invocation):
            seen.append(invocation.ref)

    class Child(dspy.Module):
        def forward(self, secret):
            with tracer.start_as_current_span("native"):
                pass
            if fail:
                raise ValueError(secret)
            return secret

    class Parent(dspy.Module):
        def forward(self, secret):
            return Child()(secret=secret)

    before = trace.get_current_span().get_span_context()
    try:
        with tracer.start_as_current_span("app") as app:
            app_context = app.get_span_context()
            with dspy.context(execution_observers=[Observer()]):
                with bridge.context(), capture_trace() as captured:
                    if fail:
                        with pytest.raises(ValueError, match="private-value"):
                            Parent()(secret="private-value")
                    else:
                        assert Parent()(secret="private-value") == "private-value"
                    assert trace.get_current_span().get_span_context() == app_context
                assert len(dspy.settings.execution_observers) == 1
        assert trace.get_current_span().get_span_context() == before
        spans = exporter.get_finished_spans()
        by_id = {span.context.span_id: span for span in spans}
        assert len(spans) == 4
        assert set(seen) == {inv.ref for inv in captured.invocations}
        root, child = captured.invocations
        assert root.parent.span_id == f"{app_context.span_id:016x}"
        for inv in captured.invocations:
            span = by_id[int(inv.ref.span_id, 16)]
            assert span.context.trace_id == int(inv.ref.trace_id, 16) == app_context.trace_id
            assert span.parent.span_id == int(inv.parent.span_id, 16)
            assert (span.start_time, span.end_time) == (inv.start_time_ns, inv.end_time_ns)
            assert span.status.status_code == (trace.StatusCode.ERROR if fail else trace.StatusCode.OK)
            assert ("private-value" in str(dict(span.attributes))) == payloads
            assert ("error.type" in span.attributes) == fail
            if payloads:
                assert json.loads(span.attributes["dspy.inputs"])["kwargs"]["secret"] == "private-value"
            else:
                assert set(span.attributes) <= {"dspy.operation", "error.type"}
        native = next(span for span in spans if span.name == "native")
        assert native.parent.span_id == int(child.ref.span_id, 16)
    finally:
        provider.shutdown()


def test_sampling_does_not_drop_optimizer_capture_in_workers():
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = TracerProvider(sampler=ALWAYS_OFF)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer(__name__)
    bridge = OTelBridge(processor)

    class Branch(dspy.Module):
        def forward(self, value):
            with tracer.start_as_current_span("native") as span:
                assert not span.get_span_context().trace_flags.sampled
            return value + 1

    try:
        with tracer.start_as_current_span("unsampled") as app:
            with bridge.context(), dspy.capture_trace() as execution:
                outputs = dspy.Parallel(num_threads=2, disable_progress_bar=True)(
                    [(Branch(), {"value": 1}), (Branch(), {"value": 8})]
                )
            assert outputs == [2, 9]
            assert len(execution.invocations) == 2 and execution.complete
            assert all(int(i.ref.trace_id, 16) == app.get_span_context().trace_id for i in execution.invocations)
        assert not exporter.get_finished_spans()
    finally:
        provider.shutdown()
