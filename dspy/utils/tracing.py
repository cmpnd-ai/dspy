"""Local tracing prototype: invocation identity, scoped capture, and consumer views.

Capture is opt-in. Nested captures are views of the same run, not independent copies.
The objects here are in-process records, not a persistence or export format.
"""

import copy
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any

from dspy.dsp.utils.settings import settings
from dspy.utils.callback_context import ACTIVE_CALL_ID

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvocationRef:
    """Portable execution identity. Callback IDs remain separate, opaque compatibility IDs."""

    trace_id: str
    span_id: str


class ExecutionObserver:
    """Experimental synchronous notifications, independent of local trace retention.

    Implementations must not block on network IO. Each notification is a detached
    snapshot; modifying it cannot change captured execution evidence.
    """

    def on_start(self, invocation):
        pass

    def on_end(self, invocation):
        pass


def tracing_enabled():
    return bool(
        settings.get("execution_traces", ())
        or settings.get("execution_observers", ())
        or settings.get("execution_tracing", False)
    )


@dataclass(frozen=True)
class ExecutionSelection:
    """Consumer-owned eligibility; never changes the observed execution."""

    invocations: frozenset[InvocationRef]


@dataclass(frozen=True)
class UnavailableValue:
    type_name: str


def snapshot(value, _ancestors=frozenset()):
    # Do not copy programs, clients, or callback closures into the execution log.
    from dspy.clients.base_lm import BaseLM
    from dspy.primitives.base_module import BaseModule

    if isinstance(value, (BaseModule, BaseLM)) or callable(value):
        return UnavailableValue(type(value).__qualname__)
    if id(value) in _ancestors:
        return UnavailableValue(type(value).__qualname__)
    ancestors = _ancestors | {id(value)}
    if isinstance(value, dict):
        return {key: snapshot(item, ancestors) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [snapshot(item, ancestors) for item in value]
        return items if isinstance(value, list) else tuple(items)
    try:
        return copy.deepcopy(value)
    except Exception:
        # Capturing an opaque user value must not break the program being observed.
        return UnavailableValue(type(value).__qualname__)


@dataclass
class Invocation:
    call_id: str
    run_id: str
    parent_id: str | None
    operation: str
    component: Any = field(repr=False)
    inputs: Any
    component_type: str = ""
    ref: InvocationRef | None = None
    parent: InvocationRef | None = None
    start_time_ns: int = field(default_factory=time.time_ns)
    end_time_ns: int | None = None
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    outputs: Any = None
    error: dict[str, str] | None = None
    data: dict[str, list[Any]] = field(default_factory=dict)
    prediction: tuple[Any, dict, Any] | None = None
    lm: Any = None
    signature: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _observers: tuple = field(default=(), repr=False)

    def detached(self):
        return replace(
            self,
            inputs=snapshot(self.inputs),
            outputs=snapshot(self.outputs),
            error=snapshot(self.error),
            data=snapshot(self.data),
            metadata=snapshot(self.metadata),
            signature=snapshot(self.signature),
            lm=snapshot(self.lm),
            prediction=None,
            component=None,
            _observers=(),
        )


def _notify(invocation, method):
    for observer in invocation._observers:
        try:
            getattr(observer, method)(invocation.detached())
        except Exception:
            logger.warning("Execution observer %s failed in %s", type(observer).__name__, method, exc_info=True)


class ExecutionTrace:
    """A captured scope. Selection changes the optimization view, never the execution log."""

    def __init__(self, run_id, parent_id, program=None, parents=()):
        self.run_id = run_id
        self.parent_id = parent_id
        self.invocations: list[Invocation] = []
        self._by_id = {}
        self._lock = threading.Lock()
        self._closed = False
        self._parents = parents
        self._excluded = set()
        self._names = (
            {id(p): name for name, p in program.named_predictors()} if hasattr(program, "named_predictors") else {}
        )

    def _append(self, invocation):
        with self._lock:
            if not self._closed:
                self.invocations.append(invocation)
                self._by_id[invocation.call_id] = invocation

    def component_name(self, invocation):
        """Optimizer name relative to this capture's program, not an invocation ID."""
        return self._names.get(id(invocation.component))

    @property
    def closed(self):
        return self._closed

    @property
    def complete(self):
        return self.closed and all(i.ended_at is not None for i in self.invocations)

    @property
    def roots(self):
        refs = {i.ref for i in self.invocations}
        return tuple(i.ref for i in self.invocations if i.parent not in refs)

    def get(self, ref):
        return next((i for i in self.invocations if i.ref == ref), None)

    def discard(self):
        """Record strategy selection in containing views, without changing invocations."""
        refs = {i.ref for i in self.invocations}
        for trace in (*self._parents, self):
            with trace._lock:
                trace._excluded.update(refs)

    def predictor_invocations(self, *, selected=True, selection=None):
        # Completion order matches the old tuple trace, including concurrent calls.
        return sorted(
            (
                i
                for i in self.invocations
                if i.prediction is not None
                and (
                    i.ref in selection.invocations
                    if selection is not None
                    else not selected or i.ref not in self._excluded
                )
            ),
            key=lambda i: i.ended_at if i.ended_at is not None else float("inf"),
        )

    def predictor_trace(self, *, selected=True, selection=None):
        return [i.prediction for i in self.predictor_invocations(selected=selected, selection=selection)]

    def to_dict(self, *, include_payloads=False):
        """Portable v1 projection. Payloads require explicit opt-in and may contain secrets.

        This is not a persistence format for live programs or a replayable History.
        Redact opted-in payloads for the destination before exporting them.
        """
        records = []
        for invocation in self.invocations:
            item = {
                "ref": asdict(invocation.ref),
                "parent": asdict(invocation.parent) if invocation.parent else None,
                "call_id": invocation.call_id,
                "operation": invocation.operation,
                "start_time_ns": invocation.start_time_ns,
                "end_time_ns": invocation.end_time_ns,
                "status": "running" if invocation.ended_at is None else "error" if invocation.error else "ok",
            }
            if include_payloads:
                item.update(
                    inputs=_portable(invocation.inputs),
                    outputs=_portable(invocation.outputs),
                    error=_portable(invocation.error),
                    data=_portable(invocation.data),
                    signature=_portable(invocation.signature),
                    metadata=_portable(invocation.metadata),
                    lm=_portable(invocation.lm),
                )
            records.append(item)
        return {"schema_version": 1, "closed": self.closed, "complete": self.complete, "invocations": records}

    def usage(self):
        """Billed usage across all attempts, independent of optimization selection."""
        from dspy.utils.usage_tracker import UsageTracker

        tracker = UsageTracker()
        for invocation in self.invocations:
            for entry in invocation.data.get("usage", []):
                tracker.add_usage(entry["model"], entry["usage"])
        return tracker.get_total_tokens()


@contextmanager
def capture_trace(program=None, *, origin=None, optimization=True, enabled=True):
    """Capture execution without changing legacy ``settings.trace`` or installing callbacks.

    Nested scopes share invocation objects and run identity. A scope excludes calls started
    after it exits, including work in abandoned workers. Await/join workers before exiting
    when a complete trace is required.
    """
    if not enabled:
        yield None
        return
    parents = settings.get("execution_traces", ())
    active = current_invocation()
    parent = active.ref if active else settings.get("execution_parent")
    trace = ExecutionTrace(
        parent.trace_id if parent else parents[-1].run_id if parents else uuid.uuid4().hex,
        ACTIVE_CALL_ID.get(),
        program,
        parents,
    )
    if origin is not None:
        # A known deepcopy edge preserves optimizer identity without guessing from signatures.
        originals = dict(origin.named_predictors())
        for name, predictor in program.named_predictors():
            for parent in parents:
                if original_name := parent._names.get(id(originals[name])):
                    parent._names[id(predictor)] = original_name
    with settings.context(execution_traces=(*parents, trace)):
        try:
            yield trace
        finally:
            with trace._lock:
                trace._closed = True
            if not optimization:
                trace.discard()


def begin_invocation(call_id, instance, operation, inputs):
    traces = settings.get("execution_traces", ())
    observers = tuple(settings.get("execution_observers", ()))
    if not traces and not observers and not settings.get("execution_tracing", False):
        return None
    active = current_invocation()
    parent = active.ref if active else settings.get("execution_parent")
    trace_id = parent.trace_id if parent else traces[-1].run_id if traces else uuid.uuid4().hex
    invocation = Invocation(
        call_id=call_id,
        run_id=trace_id,
        parent_id=ACTIVE_CALL_ID.get(),
        operation=operation,
        component=instance,
        component_type=type(instance).__qualname__,
        inputs=snapshot(inputs),
        ref=InvocationRef(trace_id, uuid.uuid4().hex[:16]),
        parent=parent,
        metadata=snapshot(settings.get("execution_metadata", {})),
        _observers=observers,
    )
    signature = inputs.get("kwargs", {}).get("signature", getattr(instance, "signature", None))
    if signature is not None:
        from dspy.signatures.signature import ensure_signature

        try:
            signature = ensure_signature(signature)
            invocation.signature = {
                "instructions": signature.instructions,
                "inputs": list(signature.input_fields),
                "outputs": list(signature.output_fields),
                "schema": signature.model_json_schema(),
            }
        except Exception:
            # Some user field types have no JSON schema. Instrumentation must not change execution.
            invocation.signature = {"unavailable": "signature schema"}
    for trace in traces:
        trace._append(invocation)
    _notify(invocation, "on_start")
    return invocation


def end_invocation(invocation, outputs, exception):
    if invocation is not None:
        invocation.outputs = snapshot(outputs)
        if exception is not None:
            invocation.error = {"type": type(exception).__qualname__, "message": str(exception)}
        invocation.ended_at = time.perf_counter()
        invocation.end_time_ns = time.time_ns()
        _notify(invocation, "on_end")


def current_invocation():
    return settings.get("active_invocation")


@contextmanager
def invocation_scope(component, operation, inputs):
    """Internal semantic boundary for plain callables such as metrics."""
    invocation = begin_invocation(uuid.uuid4().hex, component, operation, inputs)
    parent = ACTIVE_CALL_ID.get()
    if invocation is not None:
        ACTIVE_CALL_ID.set(invocation.call_id)
    outcome = {"outputs": None}
    error = None
    try:
        with settings.context(active_invocation=invocation or current_invocation()):
            yield invocation, outcome
    except BaseException as exc:
        error = exc
        raise
    finally:
        ACTIVE_CALL_ID.set(parent)
        end_invocation(invocation, outcome["outputs"], error)


def record_event(name, value):
    """Attach domain data (e.g. decision evidence or agent state) to the active invocation."""
    if invocation := current_invocation():
        invocation.data.setdefault(name, []).append(snapshot(value))


def record_prediction(predictor, inputs, prediction):
    """One predictor completion, projected to both structured and legacy traces."""
    invocation = current_invocation()
    if invocation is not None and invocation.component is predictor:
        initial = invocation.inputs.get("kwargs", {})
        captured_inputs = {key: initial[key] if key in initial else snapshot(value) for key, value in inputs.items()}
        step = (predictor, captured_inputs, snapshot(prediction))
        invocation.prediction = step
    if settings.trace is not None and settings.max_trace_size > 0:
        if len(settings.trace) >= settings.max_trace_size:
            settings.trace.pop(0)
        # Keep the legacy reference semantics, even when structured capture is enabled.
        settings.trace.append((predictor, dict(inputs), prediction))


def _portable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _portable(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _portable(value.model_dump(mode="json"))
        except Exception:
            pass
    return {"unavailable": type(value).__qualname__}
