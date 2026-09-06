from typing import Any, ClassVar
from unittest import mock

from litellm import Choices, Message, ModelResponse

import dspy
from dspy.primitives.example import Example
from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data


def test_bootstrap_trace_data():
    """Test bootstrap_trace_data function with single dspy.Predict program."""

    # Define signature for string -> int conversion
    class StringToIntSignature(dspy.Signature):
        """Convert a string number to integer"""

        text: str = dspy.InputField()
        number: int = dspy.OutputField()

    # Create program with single dspy.Predict
    program = dspy.Predict(StringToIntSignature)

    # Create dummy dataset of size 5
    dataset = [
        Example(text="one", number=1).with_inputs("text"),
        Example(text="two", number=2).with_inputs("text"),
        Example(text="three", number=3).with_inputs("text"),
        Example(text="four", number=4).with_inputs("text"),
        Example(text="five", number=5).with_inputs("text"),
    ]

    # Define exact match metric
    def exact_match_metric(example, prediction, trace=None):
        return example.number == prediction.number

    # Configure dspy
    dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())

    # Mock litellm completion responses
    # 4 successful responses and 1 that will trigger AdapterParseError
    successful_responses = [
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 1}\n```'))],
            model="openai/gpt-4o-mini",
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 2}\n```'))],
            model="openai/gpt-4o-mini",
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 3}\n```'))],
            model="openai/gpt-4o-mini",
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 4}\n```'))],
            model="openai/gpt-4o-mini",
        ),
    ]

    # Create a side effect that will trigger AdapterParseError on the 3rd call (index 2)
    def completion_side_effect(*args, **kwargs):
        call_count = completion_side_effect.call_count
        completion_side_effect.call_count += 1

        if call_count in (2, 3):
            # Return malformed responses for both structured-output mode and JSON-mode fallback.
            return ModelResponse(
                choices=[Choices(message=Message(content="This is an invalid JSON!"))],
                model="openai/gpt-4o-mini",
            )
        return successful_responses[call_count if call_count < 2 else call_count - 2]

    completion_side_effect.call_count = 0

    with mock.patch("litellm.completion", side_effect=completion_side_effect):
        # Call bootstrap_trace_data
        results = bootstrap_trace_data(
            program=program,
            dataset=dataset,
            metric=exact_match_metric,
            num_threads=1,
            raise_on_error=False,
            capture_failed_parses=True,
        )

    # Verify results
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    # Count successful and failed predictions
    successful_count = 0
    failed_count = 0

    for result in results:
        assert "example" in result
        assert "prediction" in result
        assert "trace" in result
        assert "example_ind" in result
        assert "score" in result

        if isinstance(result["prediction"], FailedPrediction):
            failed_count += 1
            # Verify failed prediction structure
            assert hasattr(result["prediction"], "completion_text")
            assert hasattr(result["prediction"], "format_reward")
            assert result["prediction"].completion_text == "This is an invalid JSON!"
        else:
            successful_count += 1
            # Verify successful prediction structure
            assert hasattr(result["prediction"], "number")

    # Verify we have the expected number of successful and failed bootstrapping
    assert successful_count == 4, f"Expected 4 successful predictions, got {successful_count}"
    assert failed_count == 1, f"Expected 1 failed prediction, got {failed_count}"

    # Verify that traces are present
    for result in results:
        assert len(result["trace"]) > 0, "Trace should not be empty"
        # Each trace entry should be a tuple of (predictor, inputs, prediction)
        for trace_entry in result["trace"]:
            assert len(trace_entry) == 3, "Trace entry should have 3 elements"


def test_bootstrap_trace_data_passes_callback_metadata(monkeypatch):
    from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

    class DummyProgram(dspy.Module):
        def forward(self, **kwargs):  # pragma: no cover - stub forward
            return dspy.Prediction()

    captured_metadata: dict[str, Any] = {}

    class DummyEvaluate:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, callback_metadata=None, **kwargs):
            captured_metadata["value"] = callback_metadata

            class _Result:
                results: ClassVar[list[Any]] = []

            return _Result()

    monkeypatch.setattr(bootstrap_trace_module, "Evaluate", DummyEvaluate)

    bootstrap_trace_module.bootstrap_trace_data(
        program=DummyProgram(),
        dataset=[],
        callback_metadata={"disable_logging": True},
    )

    assert captured_metadata["value"] == {"disable_logging": True}


def test_capture_crashes_does_not_capture_lm_errors():
    """``capture_crashes`` converts program bugs into FailedPrediction, but an LMError is
    infrastructure: repainting it as a code failure would hand the optimizer an invalid
    evaluation, so it must propagate to the evaluator's error handling instead."""
    from dspy.utils.exceptions import LMRateLimitError

    example = Example(q="x").with_inputs("q")

    class Bug(dspy.Module):
        def forward(self, **kwargs):
            raise RuntimeError("a real code failure")

    data = bootstrap_trace_data(Bug(), dataset=[example], num_threads=1, capture_crashes=True)
    assert isinstance(data[0]["prediction"], FailedPrediction)
    assert "RuntimeError" in data[0]["prediction"].completion_text

    class Flaky(dspy.Module):
        def forward(self, **kwargs):
            raise LMRateLimitError("429 from the provider")

    # raise_on_error=False mirrors the flex GEPA call site (gepa_flex_utils).
    data = bootstrap_trace_data(Flaky(), dataset=[example], num_threads=1, capture_crashes=True, raise_on_error=False)
    assert data == []  # handled by the evaluator as an error, never repainted as a FailedPrediction


def _two_required_field_signature() -> type[dspy.Signature]:
    class TwoFieldSignature(dspy.Signature):
        """Answer a question with reasoning."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        reasoning: str = dspy.OutputField()

    return TwoFieldSignature


def _partial_parse_response(missing_text: str = '{"answer": "4"}') -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=missing_text))],
        model="openai/gpt-4o-mini",
    )


def test_bootstrap_trace_data_partial_parse_retains_failed_prediction():
    """A partial parse (a parseable dict missing a required output field) must be retained
    as a ``FailedPrediction`` with an interpolated ``format_reward``, not silently dropped.

    Regression test for the ``present / expected`` list-division ``TypeError`` in the
    ``AdapterParseError`` handler: with ``raise_on_error=False`` (the production GRPO/GEPA
    configuration) the ``TypeError`` used to be swallowed and the example vanished from
    the returned ``data`` with no partial-credit signal.
    """

    program = dspy.Predict(_two_required_field_signature())
    dataset = [Example(question="What is 2+2?").with_inputs("question")]

    def exact_match_metric(example, prediction, trace=None):
        return getattr(prediction, "answer", None) == "4"

    dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())

    # The LM returns valid JSON that parses to a non-empty dict, but it is missing the
    # required `reasoning` output field. This drives the partial-parse branch where
    # `present` is a non-empty list.
    with mock.patch("litellm.completion", side_effect=lambda *a, **k: _partial_parse_response()):
        results = bootstrap_trace_data(
            program=program,
            dataset=dataset,
            metric=exact_match_metric,
            num_threads=1,
            raise_on_error=False,
            capture_failed_parses=True,
        )

    # The example must be retained instead of being silently dropped by the
    # list/list-division TypeError -> ValueError -> continue path.
    assert len(results) == 1, f"Expected 1 retained result, got {len(results)}"

    prediction = results[0]["prediction"]
    assert isinstance(prediction, FailedPrediction), (
        f"Expected a FailedPrediction to be retained, got {type(prediction).__name__}"
    )

    # 1 of 2 expected required fields present interpolates to -0.5 with the default
    # format_failure_score=-1 / failure_score=0.
    assert prediction.format_reward == -0.5
    assert isinstance(prediction.format_reward, float)
    # The reward must lie strictly between the two extremes, never at or beyond them.
    assert -1 < prediction.format_reward < 0

    # The malformed-but-parseable LM response must be captured for GRPO/GEPA training.
    assert prediction.completion_text == '{"answer": "4"}'

    # The interpolated reward must flow through `wrapped_metric` into the score that
    # GRPO/GEPA consumes.
    assert "score" in results[0]
    assert results[0]["score"] == prediction.format_reward

    # A trace entry must be recorded for the failed prediction.
    assert len(results[0]["trace"]) > 0
    for trace_entry in results[0]["trace"]:
        assert len(trace_entry) == 3
        # The trace's prediction slot carries the FailedPrediction, not a raw Prediction.
        assert trace_entry[2] is prediction


def test_bootstrap_trace_data_partial_parse_custom_scores_interpolate():
    """The partial-parse reward must interpolate using the caller-supplied
    ``failure_score`` / ``format_failure_score`` (GRPO overrides both), confirming the
    formula threads the arguments through rather than hard-coding the defaults.
    """

    program = dspy.Predict(_two_required_field_signature())
    dataset = [Example(question="What is 2+2?").with_inputs("question")]

    dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())

    with mock.patch("litellm.completion", side_effect=lambda *a, **k: _partial_parse_response()):
        results = bootstrap_trace_data(
            program=program,
            dataset=dataset,
            metric=None,
            num_threads=1,
            raise_on_error=False,
            capture_failed_parses=True,
            failure_score=1.0,
            format_failure_score=-2.0,
        )

    assert len(results) == 1
    prediction = results[0]["prediction"]
    assert isinstance(prediction, FailedPrediction)
    # 1 of 2 fields present: -2.0 + (1.0 - (-2.0)) * (1 / 2) == -0.5
    assert prediction.format_reward == -0.5
    assert -2.0 < prediction.format_reward < 1.0
