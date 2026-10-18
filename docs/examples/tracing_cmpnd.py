"""Local, executable consumer-contract example; no cmpnd dependency or network IO.

Contract inspected at cmpnd-ai/cmpnd-platform commit
fbdd2a863962f5adf97b1212ebbb5bfce104e8e8:
https://github.com/cmpnd-ai/cmpnd-platform/blob/fbdd2a863962f5adf97b1212ebbb5bfce104e8e8/docs/ingest-api.yaml
and python-sdk/cmpnd/models.py (LMUsageObservationV1, SpanBuilder).

This is NOT an SDK migration or backend integration test. The sink receives
ingestion-shaped POST paths and bodies, but must stay in memory. Production
consumers need payload redaction, bounded queues, delivery/retry policy and a
provider-specific usage normalizer. Never do network IO inside these hooks.

Usage::

    events = []
    observer = CmpndObserver(lambda path, body: events.append((path, body)))
    with dspy.context(execution_observers=[observer]):
        result = program(question="hello")  # no capture_trace or history needed
    observer.finalize()  # only after all workers have joined

Evaluate's result.executions contains each example's attempts and accepted
attempt. Join attempt.program and attempt.metric refs with observer.address(ref),
not history positions or model names. Those tuples identify roots, not every
descendant. attempt.execution is only retained with capture_execution=True.
"""

import hashlib
import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from uuid import uuid4

from dspy.utils.tracing import ExecutionObserver


def _json_value(value):
    """Destination-owned payload projection; opaque values are not repr'd."""

    def encode(item):
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        return {"unavailable": type(item).__qualname__}

    return json.loads(json.dumps(value, default=encode))


def timestamp(ns):
    seconds, nanos = divmod(ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{nanos:09d}Z"


class CmpndObserver(ExecutionObserver):
    """One bounded run, injected synchronous in-memory sink; finalize after joining.

    Canonical trace IDs are opaque: even a valid OTel 32-hex ID need not be a
    UUIDv4. Allocate a fresh cmpnd UUIDv4 and keep explicit correlation instead
    of inserting UUID hyphens into foreign identity. Keep OTel-compatible span
    IDs unchanged (the backend permits them despite the schema's UUID hint).
    """

    def __init__(self, sink):
        self.sink = sink
        self.trace_ids = {}
        self._traces = {}
        self._active = set()
        self._lock = threading.RLock()

    def address(self, ref):
        with self._lock:
            if ref.trace_id not in self.trace_ids:
                self.trace_ids[ref.trace_id] = str(uuid4())
            return {"trace_id": self.trace_ids[ref.trace_id], "span_id": ref.span_id}

    def _emit(self, trace_id, action, body):
        self.sink(f"/api/v1/traces/{trace_id}/{action}", body)

    def on_start(self, invocation):
        with self._lock:
            address = self.address(invocation.ref)
            trace_id = address["trace_id"]
            if trace_id not in self._traces:
                self._traces[trace_id] = {"end": None, "error": False}
                self._emit(trace_id, "start", {"start_time": timestamp(invocation.start_time_ns)})
            self._active.add(invocation.ref)

    def on_end(self, invocation):
        with self._lock:
            address = self.address(invocation.ref)
            trace_id = address["trace_id"]
            signature = invocation.signature or {}
            attributes = {
                "dspy.ref": json.dumps(asdict(invocation.ref)),
                "dspy.parent": json.dumps(asdict(invocation.parent) if invocation.parent else None),
                "dspy.call_id": invocation.call_id,  # legacy compatibility, never span identity
                "dspy.metadata": json.dumps(_json_value(invocation.metadata)),
                "dspy.signature": json.dumps(_json_value(signature)),
                # Completed paid work can survive a failed multi-candidate call
                # as usage events even when there is no final LMCallEvidence.
                "dspy.data": json.dumps(_json_value(invocation.data)),
            }
            span = {
                **address,
                "parent_span_id": invocation.parent.span_id if invocation.parent else None,
                "name": invocation.operation,
                "span_type": "lm_call" if invocation.operation == "lm" else "unknown",
                "start_time": timestamp(invocation.start_time_ns),
                "end_time": timestamp(invocation.end_time_ns),
                "status": "error" if invocation.error else "ok",
                "inputs": {"value": _json_value(invocation.inputs)},
                "outputs": {"value": _json_value(invocation.outputs)},
                "signature_instructions": signature.get("instructions"),
                "signature_input_fields": signature.get("inputs"),
                "signature_output_fields": signature.get("outputs"),
                "attributes": attributes,
            }
            if invocation.error:
                span.update(error_type=invocation.error["type"], error_message=invocation.error["message"])
            if invocation.lm:
                lm = invocation.lm
                # Preserve ALL evidence, including multiple responses and cached raw usage.
                attributes["dspy.lm"] = json.dumps(_json_value(lm))
                billable = lm.usage_reported and not lm.cache_hit
                counts = {
                    key: lm.usage.get(key) if billable else None
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                raw = _json_value(lm.usage_raw)
                # The wire observation requires an object; multi-response evidence is
                # explicitly wrapped, never silently reduced to the first response.
                if raw is not None and not isinstance(raw, dict):
                    raw = {"responses": raw}
                observation = {
                    "schema_version": 1,
                    "source_observation_id": invocation.ref.span_id,
                    "observed_at": span["end_time"],
                    "normalization_version": 1,
                    "usage_reported": billable,
                    "usage_raw": raw,
                    "model_requested": lm.model_requested,
                    "model_reported": lm.model_reported,
                    "service_tier_requested": lm.service_tier_requested,
                    "service_tier_effective": lm.service_tier_reported,
                    "dspy_client_cache_hit": lm.cache_hit,
                    "sdk_call_id": invocation.call_id,
                    **counts,
                }
                canonical = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                observation["source_payload_digest"] = hashlib.sha256(canonical.encode()).hexdigest()
                # DSPy cost is an estimate, NOT provider_cost (provider-origin only).
                span.update(model_name=lm.model_reported or lm.model_requested, lm_usage=observation, **counts)
                if billable and lm.cost is not None:
                    span["cost"] = lm.cost
            self._emit(trace_id, "spans", {"spans": [span]})
            state = self._traces[trace_id]
            state["end"] = max(state["end"] or 0, invocation.end_time_ns)
            state["error"] |= bool(invocation.error)
            self._active.remove(invocation.ref)

    def finalize(self):
        """Explicit scope boundary, not a guess from a root's end notification.

        Multiple roots may share a trace and workers may outlive a root. The
        caller must stop producing work and join it before calling this method.
        """
        with self._lock:
            if self._active:
                raise RuntimeError("Join all observed work before finalizing")
            for trace_id, state in self._traces.items():
                self._emit(
                    trace_id,
                    "finalize",
                    {
                        "end_time": timestamp(state["end"]),
                        "status": "error" if state["error"] else "ok",
                    },
                )
            self._traces.clear()
