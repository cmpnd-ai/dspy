from unittest.mock import patch

import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFinetune
from dspy.utils.dummies import DummyLM


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
    assert bootstrap.multitask, "Multitask should default to True"


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
    """A student predictor without an LM must fail fast, before bootstrapping.

    Previously, ``compile`` discarded the boolean returned by
    ``all_predictors_have_lms`` and only raised a per-predictor ``ValueError``
    after running the entire (expensive, real-cost) bootstrapping phase. The fix
    switches to ``assert_all_predictors_have_lms``, so compilation must fail
    before any teacher LM calls, with a clear message that names the remedy
    (``set_lm``) and warns it overwrites existing LMs.
    """
    # No global LM configured: previously compilation died inside
    # ``bootstrap_trace_data`` with a confusing "No LM is loaded" / unpacking
    # error instead of the actionable LM-assignment message.
    dspy.configure(lm=None)

    student = SimpleModule("input -> output")
    # Intentionally NOT setting LM for the student module

    bootstrap = BootstrapFinetune(metric=simple_metric)

    # If the fix regresses, ``bootstrap_trace_data`` would actually run (made
    # one of the most expensive calls in compilation); guard against that by
    # making any bootstrapping attempt fail the test loudly.
    with patch("dspy.teleprompt.bootstrap_finetune.bootstrap_trace_data") as mock_bootstrap:
        mock_bootstrap.side_effect = AssertionError("Bootstrapping must not run when a student predictor has no LM")
        with pytest.raises(ValueError, match=r"student\.set_lm\(lm\)") as excinfo:
            bootstrap.compile(student, trainset=trainset)

        # Fail-fast: the teacher LM calls that bootstrap the trace data must
        # never have been made.
        mock_bootstrap.assert_not_called()

    message = str(excinfo.value)
    # The error comes from ``assert_all_predictors_have_lms`` now, not the
    # late per-predictor loop, so its wording must match the assert helper.
    assert "does not have an LM assigned" not in message
    assert "set_lm" in message
    assert "overwrite" in message


def test_compile_fails_fast_even_with_global_lm():
    """Fail-fast must hold regardless of a configured global LM.

    With a global LM set, the student's LM-less predictors silently fall back to
    it during bootstrapping, so before the fix the wasted bootstrapping run
    completed and only the late per-predictor check surfaced the error. The fix
    must short-circuit before bootstrapping in this configuration too.
    """
    # The pre-existing bug's most-expensive failure mode is the one where a
    # global LM is configured, because bootstrapping silently succeeds.
    lm = DummyLM([{"output": "test"}])
    dspy.configure(lm=lm)

    student = SimpleModule("input -> output")
    # Intentionally NOT setting LM for the student module; the global LM would
    # be picked up during bootstrapping if the check were skipped.

    bootstrap = BootstrapFinetune(metric=simple_metric)

    with patch("dspy.teleprompt.bootstrap_finetune.bootstrap_trace_data") as mock_bootstrap:
        mock_bootstrap.side_effect = AssertionError("Bootstrapping must not run when a student predictor has no LM")
        with pytest.raises(ValueError, match=r"student\.set_lm\(lm\)"):
            bootstrap.compile(student, trainset=trainset)

        mock_bootstrap.assert_not_called()


def test_compile_names_the_predictors_missing_an_lm():
    """The error lists exactly the predictors that are missing an LM."""
    dspy.configure(lm=None)

    class TwoPredictorModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = Predict("input -> output")
            self.summarize = Predict("input -> output")

        def forward(self, **kwargs):
            return self.qa(**kwargs)

    student = TwoPredictorModule()
    # Only the first predictor has an LM set.
    student.qa.set_lm(DummyLM([{"output": "test"}]))

    bootstrap = BootstrapFinetune(metric=simple_metric)

    with patch("dspy.teleprompt.bootstrap_finetune.bootstrap_trace_data") as mock_bootstrap:
        mock_bootstrap.side_effect = AssertionError("Bootstrapping must not run when a student predictor has no LM")
        with pytest.raises(ValueError) as excinfo:
            bootstrap.compile(student, trainset=trainset)

        mock_bootstrap.assert_not_called()

    message = str(excinfo.value)
    assert "summarize" in message
    assert "'qa'" not in message
    assert "set_lm" in message


def test_compile_accepts_student_with_all_lms_set():
    """A student whose predictors all have an LM passes the pre-check.

    Bootstrapping must proceed (no early ``ValueError``) and the compilation
    must reach fine-tuning when every predictor has an LM set.
    """
    lm = DummyLM([{"output": "blue"}])
    dspy.configure(lm=lm)

    student = SimpleModule("input -> output")
    student.set_lm(lm)

    bootstrap = BootstrapFinetune(metric=simple_metric)

    # ``finetune_lms`` is the step right after the (now-still-run) bootstrapping
    # phase; mock it so the offline test does not attempt real fine-tuning.
    with patch.object(bootstrap, "finetune_lms", return_value={(lm, None): lm}):
        compiled = bootstrap.compile(student, trainset=trainset)

    assert compiled is student
    assert compiled._compiled is True


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
