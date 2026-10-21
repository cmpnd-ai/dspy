"""Metric requests, independent of tracing and caller-owned result interpretation."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

from dspy.utils.execution import Invocation

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class MetricContext:
    """The question a metric answers, with its candidate execution and feedback target.

    Results remain caller-specific. Bootstrap requests a score and applies its own
    admission policy; GEPA requests feedback about one exact predictor occurrence.
    Evidence lives on the invocation, not in a second bundle of trace/usage fields.
    None means no candidate was captured. Treat retained calls as read-only.
    """

    purpose: Literal["score", "feedback"] = "score"
    invocation: Invocation | None = None
    target: Invocation | None = None


def accepts_metric_context(metric: object) -> bool:
    try:
        candidate = getattr(metric, "forward", metric)
        if not callable(candidate):
            return False
        parameter = inspect.signature(candidate).parameters.get("metric_context")
        return parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        )
    except (TypeError, ValueError):
        return False


def call_metric(
    metric: Callable[..., ResultT], example: object, prediction: object, context: MetricContext, *,
    legacy_args: tuple[object, ...] = (), legacy_kwargs: dict[str, object] | None = None,
) -> ResultT:
    """Opt in by declaration, not **kwargs; preserve the caller's result type.

    Argument dispatch remains reflective for legacy metrics with different signatures.
    """
    if accepts_metric_context(metric):
        return metric(example, prediction, metric_context=context)
    return metric(example, prediction, *legacy_args, **(legacy_kwargs or {}))
