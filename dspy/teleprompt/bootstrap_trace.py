import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, TypedDict

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metric_context import MetricContext, call_metric
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError, LMError
from dspy.utils.execution import Invocation, capture_invocation

logger = logging.getLogger(__name__)


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[Invocation]
    invocation: Invocation
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
    capture_usage: bool = False,
    capture_execution: bool = False,
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
        prediction, invocation, _execution = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return call_metric(
            metric, example, prediction,
            MetricContext(invocation=invocation),
            legacy_args=(trace,),
        ) if metric else True

    def collect(**kwargs):
        with (
            dspy.context(trace=[], _trace_program=program),
            capture_invocation(program, recording=capture_execution) as capture,
        ):
            try:
                prediction = program(**kwargs)
                return prediction, dspy.settings.trace.copy(), capture
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                failed_call = getattr(e, "_execution_call", None)
                if failed_call is None:
                    raise ValueError("The parse error did not originate in an observed predictor call") from e

                trace = dspy.settings.trace.copy()
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score) * (present / expected),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)
                trace.append(replace(failed_call, outputs=failed_pred))

                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following "
                        "the adapter's formatting."
                    )

                return failed_pred, trace, capture
            except LMError:
                raise
            except Exception as e:
                if not capture_crashes:
                    raise
                logger.warning("Program raised on an example; capturing it as a failed prediction: %s: %s",
                               type(e).__name__, e)
                return (
                    FailedPrediction(completion_text=f"{type(e).__name__}: {e}"), dspy.settings.trace.copy(),
                    capture,
                )

    class CapturedProgram(Module):
        def forward(self, **kwargs):
            prediction, trace, capture = collect(**kwargs)
            invocation = capture.invocation
            invocation.trace = tuple(trace)
            return prediction, invocation, capture.trace if capture_execution else None

    results = evaluator(
        CapturedProgram(),
        metric=wrapped_metric,
        callback_metadata=callback_metadata,
    ).results

    data = []
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, invocation, execution = prediction
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
        data_dict = {
            "example": example, "prediction": prediction, "trace": list(invocation.trace),
            "invocation": invocation, "example_ind": example_ind,
        }
        if execution is not None:
            data_dict["execution"] = execution
        if capture_usage:
            data_dict["usage"] = invocation.usage
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data
