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


@dataclass
class Invocation:
    """One observed call, retained as typed evidence by predictor trace collectors.

    ``arguments`` contains bound callback arguments; ``inputs`` contains prepared
    predictor inputs (including defaults and hints, excluding LM/config overrides).
    Component names are predictor addresses relative to the collecting program,
    not paths through every observed module or signature identities.
    Tuple access is the migration view: (instance, inputs, outputs).
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
    def __init__(self, program=None, *, recording=True):
        self.trace_id = uuid.uuid4().hex
        self.recording = recording
        self._spans = {}
        self._names = WeakKeyDictionary()
        self._lock = threading.RLock()
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
            if invocation.kind == "predict" and exception is not None:
                failed = getattr(exception, "_execution_call", None)
                if failed is None or failed.ref.trace_id != self.trace_id:
                    exception._execution_call = invocation
            if call_id not in self._spans:
                return
            if hasattr(outputs, "toDict"):
                outputs = outputs.toDict()
            self._spans[call_id].update(
                end_time=invocation.end_time,
                outputs=json.loads(json.dumps(serialize_for_json(outputs))),
                error={"type": type(exception).__name__, "message": str(exception)} if exception else None,
            )

    @property
    def trace(self):
        with self._lock:
            return ExecutionTrace.from_dict({
                "trace_id": self.trace_id, "closed": self._closed, "spans": list(self._spans.values()),
            })

    def close(self):
        with self._lock:
            self._closed = True
            self._names.clear()


def capture_execution(program=None):
    """Capture this scope, including joined DSPy workers, independently of callbacks.

    Nested scopes are isolated records. Work finishing after scope exit cannot
    modify the sealed record; unfinished spans remain visibly unfinished.
    Optional ``program`` supplies component paths when capture begins inside it.
    """
    return _Capture(program)
