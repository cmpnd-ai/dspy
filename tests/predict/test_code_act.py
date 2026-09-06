import logging
import re
import threading
from unittest.mock import Mock

import pytest

import dspy
from dspy import Signature
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import CodeAct
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.utils import DummyLM
from tests.mock_interpreter import MockInterpreter, MockInterpreterFactory

pytestmark = pytest.mark.deno


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class StaticPredictor:
    def __init__(self, **fields):
        self.fields = fields

    def __call__(self, **kwargs):
        return dspy.Prediction(**self.fields)


class RaisingPredictor:
    def __call__(self, **kwargs):
        raise ValueError("unexpected extractor failure")


def add(a: float, b: float) -> float:
    "add two numbers"
    return a + b


def test_codeact_warns_that_rlm_is_preferred():
    with pytest.warns(
        DeprecationWarning,
        match=r"CodeAct is deprecated and will be removed in 3\.5\. RLM is the preferred replacement\.",
    ):
        CodeAct(BasicQA, tools=[add])


def test_codeact_code_generation(pooled_interpreter):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(pooled_interpreter, question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "code_output_0": '"2\\n"',
        "generated_code_0": "result = add(1,1)\nprint(result)",
    }


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")


def extract_maximum_minimum(input_list: str) -> dict[str, float]:
    numbers = list(map(float, input_list.split(",")))
    return {"maximum": max(numbers), "minimum": min(numbers)}


def test_codeact_support_multiple_fields(pooled_interpreter):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(ExtremumFinder, tools=[extract_maximum_minimum])
    res = program(pooled_interpreter, input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert res.trajectory == {
        "code_output_0": "\"{'maximum': 6.0, 'minimum': 2.0}\\n\"",
        "generated_code_0": "result = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)",
    }


def test_codeact_code_parse_failure(pooled_interpreter):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nparse(error\n```",
                "finished": False,
            },
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(pooled_interpreter, question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "generated_code_0": "parse(error",
        "observation_0": "Failed to execute the generated code: Invalid Python syntax. message: ",
        "generated_code_1": "result = add(1,1)\nprint(result)",
        "code_output_1": '"2\\n"',
    }


def test_codeact_code_execution_failure(pooled_interpreter):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nunknown+1\n```",
                "finished": False,
            },
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(pooled_interpreter, question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "generated_code_0": "unknown+1",
        "observation_0": "Failed to execute the generated code: NameError: [\"name 'unknown' is not defined\"]",
        "generated_code_1": "result = add(1,1)\nprint(result)",
        "code_output_1": '"2\\n"',
    }


def test_codeact_evaluate_creates_one_interpreter_per_example():
    tool_registration_barrier = threading.Barrier(4)

    def execute(code, variables):
        if code.startswith("def add"):
            tool_registration_barrier.wait(timeout=30)
            return ""
        return "2\n"

    factory = MockInterpreterFactory(execute_fn=execute)
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = StaticPredictor(answer="2")
    devset = [
        dspy.Example(question=f"What is 1+1? ({index})", answer="2").with_inputs("question") for index in range(4)
    ]

    result = dspy.Evaluate(
        devset=devset,
        metric=answer_exact_match,
        num_threads=4,
        display_progress=False,
    )(program)

    assert result.score == 100.0
    assert len(factory.instances) == 4
    assert len({id(interpreter) for interpreter in factory.instances}) == 4
    for interpreter in factory.instances:
        assert interpreter.call_count == 2
        with pytest.raises(CodeInterpreterError, match="shutdown"):
            interpreter.execute("print('closed')")


def test_codeact_factory_creates_fresh_interpreter_per_sequential_call():
    factory = MockInterpreterFactory(responses=["", "2\n"])
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = StaticPredictor(answer="2")

    first = program(question="What is 1+1?")
    second = program(question="What is 1+1 again?")

    assert first.answer == second.answer == "2"
    assert len(factory.instances) == 2
    assert factory.instances[0] is not factory.instances[1]
    for interpreter in factory.instances:
        with pytest.raises(CodeInterpreterError, match="shutdown"):
            interpreter.execute("print('closed')")


def test_codeact_allows_interpreter_as_signature_input():
    factory = MockInterpreterFactory(responses=["", "CPython\n"])
    program = CodeAct("interpreter -> answer", tools=[add], interpreter_factory=factory)
    program.codeact = Mock(return_value=dspy.Prediction(generated_code="print(interpreter)", finished=True))
    program.extractor = StaticPredictor(answer="CPython")

    result = program(interpreter="CPython")

    assert result.answer == "CPython"
    assert program.codeact.call_count == 1
    assert program.codeact.call_args.kwargs["interpreter"] == "CPython"


def test_codeact_rejects_keyword_interpreter_override():
    factory = MockInterpreterFactory()
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)

    with pytest.raises(TypeError, match="first positional argument"):
        program(question="What is 1+1?", interpreter=MockInterpreter())

    assert factory.instances == []


def test_codeact_rejects_removed_constructor_interpreter_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument 'interpreter'"):
        CodeAct(BasicQA, tools=[add], interpreter=MockInterpreter())


def test_codeact_does_not_shutdown_caller_owned_interpreter():
    factory = MockInterpreterFactory()
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = StaticPredictor(answer="2")
    interpreter = MockInterpreter(responses=["", "2\n"])

    try:
        result = program(interpreter, question="What is 1+1?")

        assert result.answer == "2"
        assert factory.instances == []
        assert interpreter.execute("print('still open')") == ""
    finally:
        interpreter.shutdown()


def test_codeact_shuts_down_factory_interpreter_when_extractor_raises():
    factory = MockInterpreterFactory(responses=["", "2\n"])
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = RaisingPredictor()

    with pytest.raises(ValueError, match="unexpected extractor failure"):
        program(question="What is 1+1?")

    assert len(factory.instances) == 1
    with pytest.raises(CodeInterpreterError, match="shutdown"):
        factory.instances[0].execute("print('closed')")


def test_codeact_propagates_terminal_interpreter_failure_and_shuts_down():
    factory = MockInterpreterFactory(responses=["", CodeInterpreterError("protocol corrupt")])
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)

    with pytest.raises(CodeInterpreterError, match="protocol corrupt"):
        program(question="What is 1+1?")

    assert len(factory.instances) == 1
    with pytest.raises(CodeInterpreterError, match="shutdown"):
        factory.instances[0].execute("print('closed')")


class CustomTool:
    def __call__(self, a: float, b: float) -> float:
        return a + b


def test_codeact_tool_validation():
    with pytest.raises(ValueError, match=r"CodeAct only accepts functions and not callable objects."):
        CodeAct(BasicQA, tools=[CustomTool()])


def _extract_trajectory_section(messages):
    """Return the content following the ``[[ ## trajectory ## ]]`` field header.

    Looks across both plain-string and structured ``content`` payloads (DummyLM emits a
    plain string under ``ChatAdapter``). Returns ``None`` when no trajectory header is
    present (e.g. the very first iteration still receives an empty trajectory, which the
    adapter renders as an empty section).
    """
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = re.search(r"\[\[ ## trajectory ## \]\]\n(.*)", content, re.DOTALL)
        if match:
            return match.group(1)
    return None


def test_codeact_per_iteration_trajectory_is_formatted_string(pooled_interpreter, caplog):
    """The per-iteration ``codeact`` call must pass the trajectory as the declared ``str``
    type, formatted into the multi-section representation, not as a raw dict / JSON literal.

    Guarantees:
    * No ``Type mismatch for field 'trajectory'`` warning is logged on any iteration
      (the ``codeact`` signature declares ``trajectory`` as ``type_=str``).
    * The per-iteration ``codeact`` prompt renders the trajectory as the structured per-key
      ``[[ ## generated_code_N ## ]]`` / ``[[ ## code_output_N ## ]]`` form used everywhere
      else for trajectory fields -- not a single-line JSON dict literal.
    * The per-iteration representation matches the representation the final extractor call
      produces for the same trajectory keys (cross-call consistency).
    * The end-to-end answer and returned trajectory dict are unchanged (no regression).
    """
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": False,
            },
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        res = program(pooled_interpreter, question="What is 1+1?")

    # No behavioral regression: answer and returned trajectory dict are unchanged.
    assert res.answer == "2"
    assert res.trajectory == {
        "generated_code_0": "result = add(1,1)\nprint(result)",
        "code_output_0": '"2\\n"',
        "generated_code_1": "result = add(1,1)\nprint(result)",
        "code_output_1": '"2\\n"',
    }

    # The ``trajectory`` field is declared ``type_=str``; the per-iteration call must not
    # trip Predict's input validator. With the bug, a warning is emitted on every iteration.
    trajectory_mismatch_warnings = [
        record for record in caplog.records if "Type mismatch for field 'trajectory'" in record.getMessage()
    ]
    assert trajectory_mismatch_warnings == [], (
        f"Expected no type-mismatch warnings for 'trajectory', got: {trajectory_mismatch_warnings}"
    )

    # lm.history indices: 0 = first codeact call (empty trajectory), 1 = second codeact call
    # (populated trajectory), 2 = extractor call (populated trajectory).
    per_iteration_trajectory = _extract_trajectory_section(lm.history[1]["messages"])
    assert per_iteration_trajectory is not None, "No trajectory section in the second codeact prompt"

    # Structured multi-section form, not a JSON dict literal.
    assert "[[ ## generated_code_0 ## ]]" in per_iteration_trajectory
    assert "[[ ## code_output_0 ## ]]" in per_iteration_trajectory
    assert not per_iteration_trajectory.lstrip().startswith("{")
    assert '{"generated_code_0"' not in per_iteration_trajectory

    # Cross-call consistency: the per-iteration call and the extractor call render the same
    # trajectory using the same per-key section headers (the extractor path already routes
    # through ``ReAct._format_trajectory``).
    extractor_trajectory = _extract_trajectory_section(lm.history[2]["messages"])
    assert extractor_trajectory is not None, "No trajectory section in the extractor prompt"
    assert "[[ ## generated_code_0 ## ]]" in extractor_trajectory
    assert "[[ ## code_output_0 ## ]]" in extractor_trajectory
    assert not extractor_trajectory.lstrip().startswith("{")
