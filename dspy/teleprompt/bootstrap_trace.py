import logging
from contextlib import nullcontext
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, TypedDict
from weakref import WeakKeyDictionary

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metric_context import MetricContext, accepts_metric_context, call_metric
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError, LMError
from dspy.utils.usage_tracker import track_usage

logger = logging.getLogger(__name__)


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None


def bootstrap_trace_data(
    program: Module,
    dataset: list[Example],
    metric: Callable | None = None,
    num_threads: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
    callback_metadata: dict[str, Any] | None = None,
    capture_crashes: bool = False,
    metric_purpose: str = "score",
    capture_usage: bool = False,
) -> list[TraceData]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, candidate_trace, usage = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return call_metric(
            metric, example, prediction, MetricContext(metric_purpose, trace=candidate_trace, usage=usage),
            legacy_args=(trace,),
        ) if metric else True

    # Use `object.__getattribute__` to bypass the custom hook `Module.__getattribute__` so that we avoid
    # the warning that `forward` is not accessed through `__call__`.
    original_forward = object.__getattribute__(program, "forward")
    capture_settings = {}
    if accepts_metric_context(metric) or capture_usage:
        capture_settings["_metric_predictor_names"] = WeakKeyDictionary({p: name for name, p in program.named_predictors()})
    if capture_usage:
        # Candidate totals include joined DSPy workers, not independent per-worker snapshots.
        capture_settings["_share_usage_tracker"] = True

    def patched_forward(program_to_use: Module, **kwargs):
        with dspy.context(trace=[], **capture_settings), track_usage() if capture_usage else nullcontext() as usage:
            try:
                prediction = original_forward(**kwargs)
                return prediction, dspy.settings.trace.copy(), usage.get_total_tokens() if usage else None
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature
                failed_inputs = kwargs

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                found_pred = None
                for pred in program_to_use.predictors():
                    if pred.signature == failed_signature:
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed signature: {failed_signature}")

                trace = dspy.settings.trace.copy()
                # Trace is Tuple[signature, inputs, prediction outputs]
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score) * (present / expected),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)

                trace.append(
                    (
                        found_pred,
                        failed_inputs,
                        failed_pred,
                    )
                )

                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following "
                        "the adapter's formatting."
                    )

                return failed_pred, trace, usage.get_total_tokens() if usage else None
            except LMError:
                raise
            except Exception as e:
                if not capture_crashes:
                    raise
                logger.warning("Program raised on an example; capturing it as a failed prediction: %s: %s",
                               type(e).__name__, e)
                return (
                    FailedPrediction(completion_text=f"{type(e).__name__}: {e}"), dspy.settings.trace.copy(),
                    usage.get_total_tokens() if usage else None,
                )

    program.forward = MethodType(patched_forward, program)

    try:
        results = evaluator(
            program,
            metric=wrapped_metric,
            callback_metadata=callback_metadata,
        ).results
    finally:
        program.forward = original_forward

    data = []
    predictor_names = capture_settings.get("_metric_predictor_names")
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, trace, usage = prediction
        except ValueError as ve:
            # TODO(GRPO Team): Often during GRPO bootstrapping, the LLM response does not follow dspy formatting. This
            # leads to a value error. To reproduce this issue, try Qwen/Qwen2.5-Coder-0.5B-Instruct with MATH dataset.
            # Proposal(Lakshya): We should capture the incorrectly-formatted LLM response, and store it in the trace,
            # and pass it to in the GRPO group with a high-negative user-configurable score.
            logger.warning(
                "Failed to unpack prediction and trace. This is likely due to the LLM response not following "
                "dspy formatting."
            )
            if raise_on_error:
                raise ve
            else:
                continue
        data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
        if predictor_names is not None:
            data_dict["trace_names"] = [predictor_names.get(step[0]) for step in trace]
        if capture_usage:
            data_dict["usage"] = usage
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data
