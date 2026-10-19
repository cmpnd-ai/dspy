import logging
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metric_context import MetricContext, accepts_metric_context, call_metric
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError, LMError
from dspy.utils.tracing import ExecutionTrace, capture_trace, tracing_enabled

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
    execution: ExecutionTrace
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
    capture_execution: bool = False,
) -> list[TraceData]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)
    capture = capture_execution or tracing_enabled() or accepts_metric_context(metric)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction):
        prediction, trace, execution = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward if prediction.format_reward is not None else format_failure_score
        return call_metric(
            metric, example, prediction, MetricContext("score", execution), legacy_args=(None,)
        ) if metric else True

    def run_example(**kwargs):
        with dspy.context(trace=[]), capture_trace(program, enabled=capture or capture_failed_parses) as execution:
            try:
                prediction = program(**kwargs)
            except AdapterParseError as e:
                if not capture_failed_parses:
                    raise
                failed = next(
                    (i for i in reversed(execution.invocations) if isinstance(i.component, dspy.Predict) and i.error),
                    None,
                )
                if failed is None:
                    raise
                present = len(e.parsed_result or {})
                expected = len(e.signature.output_fields)
                prediction = FailedPrediction(
                    completion_text=e.lm_response,
                    format_reward=format_failure_score + (failure_score - format_failure_score) * present / expected,
                )
                # Preserve the legacy training projection without inventing another invocation.
                failed.prediction = (failed.component, failed.inputs.get("kwargs", {}), prediction)
                dspy.settings.trace.append(failed.prediction)
                if log_format_failures:
                    logger.warning("Failed to parse output for example; retaining the failed predictor invocation.")
            except LMError:
                raise
            except Exception as e:
                if not capture_crashes:
                    raise
                logger.warning("Program raised on an example; capturing it as a failed prediction: %s: %s",
                               type(e).__name__, e)
                prediction = FailedPrediction(completion_text=f"{type(e).__name__}: {e}")
            trace = list(dspy.settings.trace)
        return prediction, trace, execution

    results = evaluator(run_example, metric=wrapped_metric, callback_metadata=callback_metadata).results

    data = []
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, trace, execution = prediction
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
            "example": example, "prediction": prediction,
            "trace": execution.predictor_trace() if capture else trace, "example_ind": example_ind,
        }
        if capture:
            data_dict["execution"] = execution
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data
