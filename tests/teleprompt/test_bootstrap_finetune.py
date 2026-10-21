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
        return {"inputs": inputs, "outputs": dict(outputs)}


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


@pytest.mark.parametrize("multitask", [False, True])
def test_finetune_routes_repeated_and_skipped_same_signature_calls(monkeypatch, multitask):
    class Program(dspy.Module):
        def __init__(self):
            signature = dspy.Signature("input -> output")
            self.first = Predict(signature)
            self.second = Predict(signature)

        def forward(self, input):
            result = self.second(input=input + ":second")
            if input == "both":
                self.first(input=input + ":first-1")
                result = self.first(input=input + ":first-2")
            return result

    values = ["both:second", "both:first-1", "both:first-2", "skip:second"]
    student = Program()
    student.set_lm(DummyLM({value: {"output": value} for value in values}))
    optimizer = BootstrapFinetune(adapter=TraceIdentityAdapter(), multitask=multitask, num_threads=2)
    submitted = {}

    def train(jobs):
        submitted.update({key[1]: job["train_data"] for key, job in jobs.items()})
        return {key: job["lm"] for key, job in jobs.items()}

    monkeypatch.setattr(optimizer, "finetune_lms", train)  # provider boundary only
    optimizer.compile(student, trainset=[Example(input=value).with_inputs("input") for value in ("both", "skip")])
    expected = {None: values} if multitask else {0: values[1:3], 1: [values[0], values[3]]}
    assert set(submitted) == set(expected)
    for key, inputs in expected.items():
        assert sorted(submitted[key], key=lambda item: item["inputs"]["input"]) == [
            {"inputs": {"input": value}, "outputs": {"output": value}} for value in sorted(inputs)
        ]
