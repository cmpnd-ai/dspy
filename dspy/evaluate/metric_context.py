"""Explicit metric purpose, with an adapter for existing metric call conventions."""

import inspect
from dataclasses import dataclass
from typing import Literal

from dspy.dsp.utils.settings import settings
from dspy.utils.tracing import ExecutionTrace, invocation_scope, tracing_enabled


@dataclass(frozen=True)
class MetricContext:
    purpose: Literal["score", "bootstrap", "feedback"] = "score"
    execution: ExecutionTrace | None = None
    component_name: str | None = None
    invocation_id: str | None = None

    @property
    def target(self):
        if self.execution is not None and self.invocation_id is not None:
            invocation = self.execution._by_id.get(self.invocation_id)
            return invocation.ref if invocation else None
        return None


def accepts_metric_context(metric):
    target = getattr(metric, "forward", metric)
    try:
        parameter = inspect.signature(target).parameters.get("metric_context")
        return parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    except (TypeError, ValueError):
        return False


def call_metric(metric, example, prediction, context, *, legacy_args=(), legacy_kwargs=None):
    """New metrics always receive context; old metrics retain the caller's previous ABI.

    Do not infer support from **kwargs: accepting arbitrary keywords is not opting into a
    different metric contract. Do not catch TypeError from execution and retry a metric.
    """

    def invoke():
        if accepts_metric_context(metric):
            return metric(example, prediction, metric_context=context)
        return metric(example, prediction, *legacy_args, **(legacy_kwargs or {}))

    if not tracing_enabled():
        return invoke()
    metadata = {
        **settings.get("execution_metadata", {}),
        "metric_purpose": context.purpose,
        "candidate": context.execution.roots if context.execution else (),
        "target": context.target,
        "component_name": context.component_name,
    }
    with settings.context(execution_metadata=metadata):
        with invocation_scope(metric, "metric", {"example": example, "prediction": prediction}) as (_, outcome):
            result = invoke()
            outcome["outputs"] = result
            return result
