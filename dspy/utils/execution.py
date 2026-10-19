"""Opt-in execution records built at DSPy's existing callback boundaries.

Records are detached observations, not replay state or demonstration selection.
Callbacks and records share invocation IDs. No exporter or storage is required.
"""

import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from dspy.adapters.utils import serialize_for_json
from dspy.dsp.utils.settings import settings
from dspy.utils.callback_context import ACTIVE_CALL_ID


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
    def __init__(self, program=None):
        self.trace_id = uuid.uuid4().hex
        self._spans = {}
        self._names = WeakKeyDictionary()
        self._lock = threading.RLock()
        self._closed = False
        if program is not None:
            self._register(program)

    def _register(self, program):
        for name, module in program.named_sub_modules():
            self._names[module] = name.removeprefix("self.")

    def inherit_names(self, original, copied):
        """A retry wrapper knows the provenance of its copy; a signature does not."""
        with self._lock:
            originals = dict(original.named_sub_modules())
            for name, module in copied.named_sub_modules():
                path = self._names.get(originals[name])
                if path is not None:
                    self._names[module] = path

    def start(self, call_id, instance, kind, inputs):
        from dspy.predict.predict import Predict

        with self._lock:
            if self._closed:
                return
            if not self._spans and hasattr(instance, "named_sub_modules") and not self._names:
                self._register(instance)
            parent = ACTIVE_CALL_ID.get()
            self._spans[call_id] = {
                "span_id": call_id,
                "parent_span_id": parent if parent in self._spans else None,
                "kind": "predict" if isinstance(instance, Predict) else kind,
                "name": instance.name if kind == "tool" else type(instance).__qualname__,
                "component_name": self._names.get(instance) if hasattr(instance, "named_sub_modules") else None,
                "start_time": time.time(),
                "end_time": None,
                "inputs": json.loads(json.dumps(serialize_for_json(inputs))),
                "outputs": None,
                "error": None,
            }

    def end(self, call_id, outputs, exception):
        with self._lock:
            if self._closed or call_id not in self._spans:
                return
            if exception is not None and self._spans[call_id]["kind"] == "predict":
                if getattr(exception, "_execution_ref", None) is None:
                    exception._execution_ref = ExecutionRef(self.trace_id, call_id)
            if hasattr(outputs, "toDict"):
                outputs = outputs.toDict()
            self._spans[call_id].update(
                end_time=time.time(),
                outputs=json.loads(json.dumps(serialize_for_json(outputs))),
                error={"type": type(exception).__name__, "message": str(exception)} if exception else None,
            )

    def predictor_ref(self):
        call_id = ACTIVE_CALL_ID.get()
        with self._lock:
            if not self._closed and call_id in self._spans:
                return ExecutionRef(self.trace_id, call_id)
        return None

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


@contextmanager
def capture_execution(program=None):
    """Capture this scope, including joined DSPy workers, independently of callbacks.

    Nested scopes are isolated records. Work finishing after scope exit cannot
    modify the sealed record; unfinished spans remain visibly unfinished.
    Optional ``program`` supplies component paths when capture begins inside it.
    """
    capture = _Capture(program)
    with settings.context(_execution_capture=capture):
        try:
            yield capture
        finally:
            capture.close()
