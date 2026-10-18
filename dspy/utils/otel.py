"""Optional completed-span bridge for the local execution tracing prototype.

Install ``opentelemetry-sdk`` separately. Pass a public SDK SpanProcessor (for
example BatchSpanProcessor) to OTelBridge, then use ``with bridge.context():``.
The caller owns processor flushing/shutdown. This bridge delivers completed
ReadableSpans to on_end, not live SDK Spans to on_start; processors requiring
live-span mutation are not supported. No SDK tracer or second ID tree is used.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar

from dspy.dsp.utils.settings import settings
from dspy.utils.tracing import ExecutionObserver, InvocationRef, _portable


class OTelBridge(ExecutionObserver):
    """Export canonical DSPy spans, with sensitive payloads disabled by default.

    ``include_payloads=True`` exports inputs, outputs, error messages, signature,
    metadata, domain data and LM evidence as JSON. Redact these at the destination.
    Ambient sampling flags and trace state are preserved; standalone spans are
    sampled. Native OTel spans inside a DSPy invocation inherit its canonical ID.
    """

    def __init__(self, processor, *, include_payloads=False, resource=None):
        # Keep even importing this module safe without the optional SDK.
        from opentelemetry import context, trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope

        self.processor = processor
        self.include_payloads = include_payloads
        self.resource = resource if resource is not None else Resource({})
        self._context = context
        self._trace = trace
        self._readable_span = ReadableSpan
        self._scope = InstrumentationScope("dspy.utils.otel")
        self._active = ContextVar(f"dspy_otel_{id(self)}", default=None)

    @contextmanager
    def context(self):
        """Join the ambient OTel span and append this observer for this scope.

        Enter before the root DSPy call. Existing active DSPy invocations retain
        their canonical parentage, as do scopes without a valid ambient span.
        """
        ambient = self._trace.get_current_span().get_span_context()
        observers = tuple(settings.get("execution_observers", ()))
        overrides = {"execution_observers": observers if self in observers else (*observers, self)}
        if ambient.is_valid:
            overrides["execution_parent"] = InvocationRef(f"{ambient.trace_id:032x}", f"{ambient.span_id:016x}")
            # DSPy workers propagate settings, not arbitrary library ContextVars.
            overrides["execution_otel_parent"] = ambient
        with settings.context(**overrides):
            yield self

    def on_start(self, invocation):
        trace = self._trace
        ambient = trace.get_current_span().get_span_context()
        if not ambient.is_valid:
            ambient = settings.get("execution_otel_parent", ambient)
        same_trace = ambient.trace_id == int(invocation.ref.trace_id, 16)
        flags = ambient.trace_flags if same_trace else trace.TraceFlags(trace.TraceFlags.SAMPLED)
        state = ambient.trace_state if same_trace else trace.TraceState()
        span_context = trace.SpanContext(
            int(invocation.ref.trace_id, 16),
            int(invocation.ref.span_id, 16),
            is_remote=False,
            trace_flags=flags,
            trace_state=state,
        )
        parent = None
        if invocation.parent:
            parent = trace.SpanContext(
                int(invocation.parent.trace_id, 16),
                int(invocation.parent.span_id, 16),
                is_remote=False,
                trace_flags=flags,
                trace_state=state,
            )
        token = self._context.attach(trace.set_span_in_context(trace.NonRecordingSpan(span_context)))
        active = dict(self._active.get() or {})
        active[invocation.ref] = (token, span_context, parent)
        self._active.set(active)

    def on_end(self, invocation):
        active = dict(self._active.get() or {})
        entry = active.pop(invocation.ref, None)
        if entry is None:
            return
        self._active.set(active)
        token, span_context, parent = entry
        # Restore before calling user processors, including when export fails.
        self._context.detach(token)
        if not span_context.trace_flags.sampled:
            return
        trace = self._trace
        attributes = {"dspy.operation": invocation.operation}
        if invocation.error:
            attributes["error.type"] = invocation.error["type"]
        if self.include_payloads:
            for name in ("inputs", "outputs", "error", "signature", "metadata", "data", "lm"):
                attributes[f"dspy.{name}"] = json.dumps(_portable(getattr(invocation, name)))
        self.processor.on_end(
            self._readable_span(
                name=invocation.operation,
                context=span_context,
                parent=parent,
                resource=self.resource,
                instrumentation_scope=self._scope,
                start_time=invocation.start_time_ns,
                end_time=invocation.end_time_ns,
                attributes=attributes,
                status=trace.Status(trace.StatusCode.ERROR if invocation.error else trace.StatusCode.OK),
            )
        )
