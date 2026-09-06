from unittest.mock import patch

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFinetune
from dspy.teleprompt.bootstrap_trace import FailedPrediction
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError


# Define a simple metric function for testing
def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


examples = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
]
trainset = [examples[0]]


class TraceIdentityAdapter(dspy.ChatAdapter):
    def format_finetune_data(self, signature, demos, inputs, outputs):
        return {"inputs": inputs, "outputs": outputs}


def make_two_predictor_trace():
    return [
        {
            "trace": [
                (Predict("input -> output"), {"input": "predictor-0"}, {"output": "zero"}),
                (Predict("input -> output"), {"input": "predictor-1"}, {"output": "one"}),
            ]
        }
    ]


def test_bootstrap_finetune_initialization():
    """Test BootstrapFinetune initialization with various parameters."""
    bootstrap = BootstrapFinetune(metric=simple_metric)
    assert bootstrap.metric == simple_metric, "Metric not correctly initialized"
    assert bootstrap.multitask == True, "Multitask should default to True"


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_compile_with_predict_instances():
    """Test BootstrapFinetune compilation with Predict instances."""
    # Create SimpleModule instances for student and teacher
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM([{"output": "blue"}, {"output": "Ring-ding-ding-ding-dingeringeding!"}])
    dspy.configure(lm=lm)

    # Set LM for both student and teacher
    student.set_lm(lm)
    teacher.set_lm(lm)

    bootstrap = BootstrapFinetune(metric=simple_metric)

    # Mock the fine-tuning process since DummyLM doesn't support it
    with patch.object(bootstrap, "finetune_lms") as mock_finetune:
        mock_finetune.return_value = {(lm, None): lm}
        compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

        assert compiled_student is not None, "Failed to compile student"
        assert hasattr(compiled_student, "_compiled") and compiled_student._compiled, "Student compilation flag not set"

        mock_finetune.assert_called_once()


def test_error_handling_missing_lm():
    """Test error handling when predictor doesn't have an LM assigned."""

    lm = DummyLM([{"output": "test"}])
    dspy.configure(lm=lm)

    student = SimpleModule("input -> output")
    # Intentionally NOT setting LM for the student module

    bootstrap = BootstrapFinetune(metric=simple_metric)

    # This should raise ValueError about missing LM and hint to use set_lm
    try:
        bootstrap.compile(student, trainset=trainset)
        assert False, "Should have raised ValueError for missing LM"
    except ValueError as e:
        assert "does not have an LM assigned" in str(e)
        assert "set_lm" in str(e)


def test_prepare_finetune_data_filters_to_requested_predictor():
    bootstrap = BootstrapFinetune(adapter=TraceIdentityAdapter(), exclude_demos=True)

    data, _ = bootstrap._prepare_finetune_data(
        trace_data=make_two_predictor_trace(),
        lm=DummyLM([]),
        pred_ind=1,
    )

    assert data == [{"inputs": {"input": "predictor-1"}, "outputs": {"output": "one"}}]


def test_prepare_finetune_data_includes_all_predictors_without_filter():
    bootstrap = BootstrapFinetune(adapter=TraceIdentityAdapter(), exclude_demos=True)

    data, _ = bootstrap._prepare_finetune_data(
        trace_data=make_two_predictor_trace(),
        lm=DummyLM([]),
        pred_ind=None,
    )

    assert sorted(data, key=lambda item: item["inputs"]["input"]) == [
        {"inputs": {"input": "predictor-0"}, "outputs": {"output": "zero"}},
        {"inputs": {"input": "predictor-1"}, "outputs": {"output": "one"}},
    ]


def make_trace_with_failed_prediction():
    """Step 0 is a valid prediction; step 1 is a FailedPrediction (unparseable LM response)."""
    return [
        {
            "trace": [
                (Predict("input -> output"), {"input": "predictor-0"}, {"output": "zero"}),
                (
                    Predict("input -> output"),
                    {"input": "predictor-1"},
                    FailedPrediction(completion_text="garbage", format_reward=-1),
                ),
            ]
        }
    ]


def test_prepare_finetune_data_skips_failed_prediction():
    """A FailedPrediction trace step is skipped instead of being forwarded to format_finetune_data."""
    bootstrap = BootstrapFinetune(adapter=TraceIdentityAdapter(), exclude_demos=True)

    data, _ = bootstrap._prepare_finetune_data(
        trace_data=make_trace_with_failed_prediction(),
        lm=DummyLM([]),
        pred_ind=None,
    )

    assert data == [{"inputs": {"input": "predictor-0"}, "outputs": {"output": "zero"}}]


def test_prepare_finetune_data_skips_failed_prediction_with_metric():
    """The metric score filter retains failed entries (truthy -1 score); the FailedPrediction
    guard must still prevent them from reaching format_finetune_data."""
    trace = make_trace_with_failed_prediction()
    trace[0]["score"] = -1  # truthy, so it survives the `if d["score"]` filter
    bootstrap = BootstrapFinetune(metric=simple_metric, adapter=TraceIdentityAdapter(), exclude_demos=True)

    data, _ = bootstrap._prepare_finetune_data(
        trace_data=trace,
        lm=DummyLM([]),
        pred_ind=None,
    )

    assert data == [{"inputs": {"input": "predictor-0"}, "outputs": {"output": "zero"}}]


def test_prepare_finetune_data_does_not_crash_with_default_chat_adapter():
    """With the default ChatAdapter (the crash site), a FailedPrediction must not reach
    format_finetune_data, which would otherwise raise AttributeError by calling outputs.get(...)."""
    bootstrap = BootstrapFinetune(adapter=dspy.ChatAdapter(), exclude_demos=True)

    data, _ = bootstrap._prepare_finetune_data(
        trace_data=make_trace_with_failed_prediction(),
        lm=DummyLM([]),
        pred_ind=None,
    )

    # The valid step is formatted into chat messages; the FailedPrediction is skipped.
    assert len(data) == 1
    assert "messages" in data[0]


def _make_module_raising_parse_error(signature, lm, bad_input):
    """Build a SimpleModule whose forward raises AdapterParseError for `bad_input`
    and otherwise delegates to its predictor (mirroring an unparseable LM response)."""
    module = SimpleModule(signature)
    module.set_lm(lm)
    predictor = module.predictor

    def forward(**kwargs):
        if kwargs.get("input") == bad_input:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=predictor.signature,
                lm_response="this is total garbage that does not parse at all",
                parsed_result=None,
            )
        return predictor(**kwargs)

    module.forward = forward
    return module


def test_compile_does_not_crash_when_all_bootstrapped_steps_are_failed_predictions():
    """BootstrapFinetune.compile must not crash when every bootstrapped trace step is a
    FailedPrediction (the end-to-end reproduction of the reported bug)."""
    lm = DummyLM([{"output": "blue"}])
    dspy.configure(lm=lm)
    student = _make_module_raising_parse_error("input -> output", lm, examples[0]["input"])

    bootstrap = BootstrapFinetune(metric=simple_metric)

    with patch.object(bootstrap, "finetune_lms") as mock_finetune:
        mock_finetune.return_value = {(lm, None): lm}
        compiled = bootstrap.compile(student, trainset=[examples[0]])

    assert getattr(compiled, "_compiled", False)
    mock_finetune.assert_called_once()
    # All trace steps were FailedPredictions and were skipped -> 0 training points.
    finetune_dict = mock_finetune.call_args[0][0]
    train_data = next(iter(finetune_dict.values()))["train_data"]
    assert train_data == []
