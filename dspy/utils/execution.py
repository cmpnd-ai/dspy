"""Typed invocation observations and opt-in detached execution records.

Callbacks, predictor tuples, and JSON records project the same invocation lifecycle.
Consumers still own retention and attempt selection. No exporter or storage is required.
"""

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakKeyDictionary

from dspy.adapters.utils import serialize_for_json
from dspy.dsp.utils.settings import settings
from dspy.utils.usage_tracker import UsageTracker


@dataclass
class Invocation:
    """One observed call, retained as typed evidence by predictor trace collectors.

    ``arguments`` contains bound callback arguments; ``inputs`` contains prepared
    predictor inputs (including defaults and hints, excluding LM/config overrides).
    Component names are predictor addresses relative to the collecting program,
    not paths through every observed module or signature identities.
    Tuple access is the migration view: (instance, inputs, outputs).

    Captured roots expose all observed ``children`` and a separate ``trace`` of
    selected predictor calls. ``usage`` totals reported non-cached LM work across
    the subtree; ``own_usage`` belongs only to this call. Neither is an invoice.
    Ordinary tuple collection need not retain a subtree (``children=None``).
    """

    instance: Any
    kind: str
    arguments: dict
    parent_id: str | None
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    outputs: Any = None
    exception: BaseException | None = None
    inputs: dict | None = None
    component_name: str | None = None
    ref: "ExecutionRef | None" = None
    children: tuple["Invocation", ...] | None = None
    trace: tuple["Invocation", ...] | None = None
    usage: dict | None = None
    own_usage: dict = field(default_factory=dict)
    cache_hit: bool | None = None
    cost: float | None = None
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def descendants(self):
        """Walk observed descendants, refusing to confuse uncaptured work with no work."""
        if self.children is None:
            raise ValueError("This invocation's subtree was not retained")
        for child in self.children:
            yield child
            yield from child.descendants()

    def get_total_lm_calls(self) -> int:
        """Count observed LM invocations, including this call, cache hits and failures.

        Requires a retained subtree. Provider retries and candidate fan-out inside
        one LM invocation are not separate calls. A running tree is only a snapshot.
        """
        return (self.kind == "lm") + sum(call.kind == "lm" for call in self.descendants())

    def get_total_cost(self) -> float | None:
        """Sum estimated USD cost for this call and its retained subtree.

        Cache hits cost zero; any LM call with unknown cost makes the total None.
        No LM calls means zero. Includes all observed attempts, not only selected
        predictor evidence. A running tree is only a snapshot, not a final bill.
        """
        costs = [call.cost for call in self.descendants() if call.kind == "lm"]
        if self.kind == "lm":
            costs.append(self.cost)
        return sum(costs, 0.0) if all(cost is not None for cost in costs) else None

    def _add_usage(self, usage):
        with self._lock:
            if self.usage is not None:
                for model, counts in usage.items():
                    self.usage[model] = UsageTracker._merge_usage_entries(self.usage.get(model), counts)

    def __iter__(self):
        return iter((self.instance, self.inputs, self.outputs))

    def __getitem__(self, index):
        return (self.instance, self.inputs, self.outputs)[index]

    def __len__(self):
        return 3


@dataclass(frozen=True)
class ExecutionRef:
    """An execution, an invocation, or a JSON Pointer into that invocation's output."""

    trace_id: str
    span_id: str | None = None
    output_path: str | None = None


@dataclass(frozen=True)
class Feedback:
    """A separate annotation, not a metric result or an execution event.

    Consumers own channels, reviewer identity, persistence, and revision policy.
    """

    target: ExecutionRef
    content: str


@dataclass(frozen=True)
class ExecutionTrace:
    """A detached JSON record. Reading it returns copies, never live program objects.

    Payloads use DSPy's JSON conversion, which can be lossy for custom objects.
    A running snapshot can contain unfinished spans. This is not a replay artifact.
    """

    _json: str

    @classmethod
    def from_dict(cls, data):
        return cls(json.dumps(data))

    def to_dict(self):
        return json.loads(self._json)

    @property
    def trace_id(self):
        return self.to_dict()["trace_id"]

    @property
    def spans(self):
        return tuple(self.to_dict()["spans"])

    def span(self, span_id):
        for span in self.spans:
            if span["span_id"] == span_id:
                return span
        raise KeyError(span_id)

    def ref(self, span_id=None, *, output_path=None):
        ref = ExecutionRef(self.trace_id, span_id, output_path)
        self.resolve(ref)
        return ref

    def resolve(self, ref: ExecutionRef):
        """Resolve within this record, refusing foreign IDs and invalid JSON Pointers.

        Output targets require an explicit completed invocation, even for roots:
        a capture scope can have several roots, so choosing one would be guessing.
        """
        if ref.trace_id != self.trace_id:
            raise ValueError("Feedback target belongs to another execution")
        if ref.span_id is None:
            if ref.output_path is not None:
                raise ValueError("An output target requires an invocation")
            return self.to_dict()
        span = self.span(ref.span_id)
        if ref.output_path is None:
            return span
        path = ref.output_path
        if (path and not path.startswith("/")) or re.search(r"~(?![01])", path):
            raise ValueError("Invalid JSON Pointer")
        if span["end_time"] is None:
            raise ValueError("An unfinished invocation has no recorded output")
        value = span["outputs"]
        for token in path.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", token) or int(token) >= len(value):
                    raise KeyError(path)
                value = value[int(token)]
            elif isinstance(value, dict):
                value = value[token]
            else:
                raise KeyError(path)
        return value

    def annotate(self, content: str, *, span_id=None, output_path=None):
        return Feedback(self.ref(span_id, output_path=output_path), content)


class _Capture:
    def __init__(self, program=None, *, recording=True, retain=True, parent=None):
        self.trace_id = parent.trace_id if parent is not None else uuid.uuid4().hex
        self.recording = recording
        self.retain = retain
        self.roots = ()
        self._active = set()
        self._parent = parent
        self._spans = {}
        self._names = parent._names if parent is not None else WeakKeyDictionary()
        self._lock = parent._lock if parent is not None else threading.RLock()
        self._closed = False
        if program is not None:
            self._register(program)

    def __enter__(self):
        self._context = settings.context(_execution_capture=self)
        self._context.__enter__()
        return self

    def __exit__(self, *exc):
        self.close()
        return self._context.__exit__(*exc)

    @property
    def invocation(self):
        """The candidate starts first; observer hooks may create additional sibling roots."""
        return self.roots[0]

    def _register(self, program):
        # Component names are optimizer addresses, not paths through arbitrary
        # runtime objects (notably accumulated LM history).
        self._names[program] = None
        for name, predictor in program.named_predictors():
            self._names[predictor] = name

    def inherit_names(self, original, copied):
        """A retry wrapper knows the provenance of its copy; a signature does not."""
        with self._lock:
            originals = dict(original.named_predictors())
            for name, predictor in copied.named_predictors():
                path = self._names.get(originals[name])
                if path is not None:
                    self._names[predictor] = path

    def start(self, invocation):
        instance = invocation.instance
        with self._lock:
            if self._closed:
                return
            if self._parent is not None:
                self._parent.start(invocation)
            if self.retain and invocation.parent_id not in self._active:
                self.roots += (invocation,)
            self._active.add(invocation.call_id)
            # A collector declares its optimizer-relative root, even inside an
            # outer observation scope. Copies inherit these names explicitly.
            if instance is settings.get("_trace_program"):
                self._register(instance)
            elif not self._names and hasattr(instance, "named_predictors"):
                self._register(instance)
            invocation.component_name = self._names.get(instance) if invocation.kind == "predict" else None
            invocation.ref = ExecutionRef(self.trace_id, invocation.call_id)
            if not self.recording:
                return
            parent = invocation.parent_id
            self._spans[invocation.call_id] = {
                "span_id": invocation.call_id,
                "parent_span_id": parent if parent in self._spans else None,
                "kind": invocation.kind,
                "name": instance.name if invocation.kind == "tool" else type(instance).__qualname__,
                "component_name": invocation.component_name,
                "start_time": invocation.start_time,
                "end_time": None,
                "inputs": json.loads(json.dumps(serialize_for_json(invocation.arguments))),
                "outputs": None,
                "error": None,
            }

    def end(self, invocation):
        call_id, outputs, exception = invocation.call_id, invocation.outputs, invocation.exception
        with self._lock:
            if self._closed:
                return
            if self._parent is not None:
                self._parent.end(invocation)
            self._active.discard(call_id)
            if invocation.kind == "predict" and exception is not None:
                failed = getattr(exception, "_execution_call", None)
                if failed is None or failed.ref.trace_id != self.trace_id:
                    exception._execution_call = invocation
            if call_id not in self._spans:
                return
            if hasattr(outputs, "toDict"):
                outputs = outputs.toDict()
            self._spans[call_id].update(json.loads(json.dumps(serialize_for_json({
                "end_time": invocation.end_time,
                "outputs": outputs,
                "error": {"type": type(exception).__name__, "message": str(exception)} if exception else None,
                "usage": invocation.usage,
                "own_usage": invocation.own_usage,
                "cache_hit": invocation.cache_hit,
                "cost": invocation.cost,
            }))))

    @property
    def trace(self):
        with self._lock:
            return ExecutionTrace.from_dict({
                "trace_id": self.trace_id, "closed": self._closed, "spans": list(self._spans.values()),
            })

    def close(self):
        with self._lock:
            for root in self.roots:
                ids = {root.call_id, *(call.call_id for call in root.descendants())}
                root.trace = None if settings.trace is None else tuple(
                    call for call in settings.trace if isinstance(call, Invocation) and call.call_id in ids
                )
            self._closed = True
            self._active.clear()
            if self._parent is None:
                self._names.clear()


def capture_execution(program=None):
    """Capture this scope, including joined DSPy workers, independently of callbacks.

    Nested scopes are isolated records. Work finishing after scope exit cannot
    modify the sealed record; unfinished spans remain visibly unfinished.
    Optional ``program`` supplies component paths when capture begins inside it.
    ``roots`` retains live invocation trees; ``trace`` is the detached JSON view.
    Retaining the handle therefore retains live inputs, outputs, and program objects.
    """
    return _Capture(program)


def capture_invocation(program=None, *, recording=False):
    """Internal candidate view: retain calls without hiding them from an outer recorder."""
    return _Capture(program, recording=recording, parent=settings.get("_execution_capture"))


def record_usage(model, usage, *, cache_hit=False, cost=None):
    """Attach reported usage and estimated USD cost at the observed LM boundary."""
    from dspy.utils.callback_context import ACTIVE_INVOCATION

    call = ACTIVE_INVOCATION.get()
    if call is None or call.kind != "lm":
        return
    if call.cache_hit is None:
        call.cost = 0.0
    if not cache_hit:
        call.cost = call.cost + cost if call.cost is not None and cost is not None else None
    call.cache_hit = cache_hit
    if not cache_hit and usage:
        counts = UsageTracker._flatten_usage_entry(usage)
        call.own_usage[model] = UsageTracker._merge_usage_entries(call.own_usage.get(model), counts)
        call._add_usage({model: counts})
