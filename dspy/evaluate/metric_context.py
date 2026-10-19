"""Metric requests, independent of tracing and caller-owned result interpretation."""

import inspect
from dataclasses import dataclass
from typing import Literal

from dspy.utils.execution import ExecutionRef, ExecutionTrace, Invocation


@dataclass(frozen=True)
class MetricContext:
    """Why a metric is called and the evidence its caller already has.

    Trace membership is snapshotted, not the predictor/input/output objects. A target
    is an occurrence index in that trace, not a component or global execution ID.
    None means evidence is unavailable; an empty trace/usage means none was recorded.
    Usage contains reported non-cached totals by model, not an invoice.
    """

    purpose: Literal["score", "accept", "feedback"] = "score"
    trace: tuple[Invocation, ...] | None = None
    component_name: str | None = None
    target: int | None = None
    usage: dict | None = None
    execution: ExecutionTrace | None = None

    def __post_init__(self):
        if self.trace is not None:
            object.__setattr__(self, "trace", tuple(self.trace))

    @property
    def target_ref(self) -> ExecutionRef | None:
        """Durable identity of the selected predictor occurrence, when recorded."""
        if self.execution is None or self.trace is None or self.target is None:
            return None
        ref = self.trace[self.target].ref
        if ref is not None and ref.trace_id == self.execution.trace_id:
            return ref
        return None


def accepts_metric_context(metric):
    try:
        parameter = inspect.signature(getattr(metric, "forward", metric)).parameters.get("metric_context")
        return parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        )
    except (TypeError, ValueError):
        return False


def call_metric(metric, example, prediction, context, *, legacy_args=(), legacy_kwargs=None):
    """Opt in by declaration, not **kwargs; call once and return the result unchanged."""
    if accepts_metric_context(metric):
        return metric(example, prediction, metric_context=context)
    return metric(example, prediction, *legacy_args, **(legacy_kwargs or {}))
