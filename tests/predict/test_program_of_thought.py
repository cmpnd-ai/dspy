import threading
from unittest.mock import Mock, patch

import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.evaluate.metrics import answer_exact_match
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.utils import DummyLM
from tests.mock_interpreter import MockInterpreter, MockInterpreterFactory


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class StaticPredictor:
    def __init__(self, **fields):
        self.fields = fields

    def __call__(self, **kwargs):
        return dspy.Prediction(**self.fields)


class RecordingPythonInterpreterFactory:
    def __init__(self, parties: int):
        self.instances = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(parties)

    def __call__(self):
        interpreter = PythonInterpreter()
        with self._lock:
            self.instances.append(interpreter)
        self._barrier.wait(timeout=30)
        return interpreter


def test_pot_warns_that_rlm_is_preferred():
    with pytest.warns(
        DeprecationWarning,
        match=r"ProgramOfThought is deprecated and will be removed in 3\.5\. RLM is the preferred replacement\.",
    ):
        ProgramOfThought(BasicQA)


@pytest.mark.deno
def test_pot_code_generation(pooled_interpreter):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```",
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(pooled_interpreter, question="What is 1+1?")
    assert res.answer == "2"


# This test ensures the old finetuned saved models still work
@pytest.mark.deno
def test_old_style_pot(pooled_interpreter):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(pooled_interpreter, question="What is 1+1?")
    assert res.answer == "2"


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")


@pytest.mark.deno
def test_pot_support_multiple_fields(pooled_interpreter):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nmaximum = 6\nminimum = 2\nSUBMIT({'maximum': maximum, 'minimum': minimum})\n```",
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(ExtremumFinder)
    res = pot(pooled_interpreter, input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"


@pytest.mark.deno
def test_pot_code_generation_with_one_error(pooled_interpreter):
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```",
            },
            {
                "reasoning": "Reason_B",
                "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```",
            },
            {"reasoning": "Reason_C", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(pooled_interpreter, question="What is 1+1?")
    assert res.answer == "2"


@pytest.mark.deno
def test_pot_evaluate_creates_one_interpreter_per_example():
    factory = RecordingPythonInterpreterFactory(parties=4)
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="SUBMIT({'answer': 2})")
    pot.generate_output = StaticPredictor(answer="2")
    devset = [
        dspy.Example(question=f"What is 1+1? ({index})", answer="2").with_inputs("question") for index in range(4)
    ]

    result = dspy.Evaluate(
        devset=devset,
        metric=answer_exact_match,
        num_threads=4,
        display_progress=False,
    )(pot)

    assert result.score == 100.0
    assert len(factory.instances) == 4
    assert len({id(interpreter) for interpreter in factory.instances}) == 4
    assert all(interpreter.deno_process is None for interpreter in factory.instances)


def test_pot_factory_creates_fresh_interpreter_per_sequential_call():
    factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "2"})])
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="SUBMIT({'answer': 2})")
    pot.generate_output = StaticPredictor(answer="2")

    first = pot(question="What is 1+1?")
    second = pot(question="What is 1+1 again?")

    assert first.answer == second.answer == "2"
    assert len(factory.instances) == 2
    assert factory.instances[0] is not factory.instances[1]
    for interpreter in factory.instances:
        with pytest.raises(CodeInterpreterError, match="shutdown"):
            interpreter.execute("print('closed')")


def test_pot_allows_interpreter_as_signature_input():
    factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "CPython"})])
    pot = ProgramOfThought("interpreter -> answer", interpreter_factory=factory)
    pot.code_generate = Mock(return_value=dspy.Prediction(generated_code="SUBMIT({'answer': interpreter})"))
    pot.generate_output = StaticPredictor(answer="CPython")

    result = pot(interpreter="CPython")

    assert result.answer == "CPython"
    pot.code_generate.assert_called_once_with(interpreter="CPython")


def test_pot_rejects_keyword_interpreter_override():
    factory = MockInterpreterFactory()
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)

    with pytest.raises(TypeError, match="first positional argument"):
        pot(question="What is 1+1?", interpreter=MockInterpreter())

    assert factory.instances == []


def test_pot_rejects_removed_constructor_interpreter_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument 'interpreter'"):
        ProgramOfThought(BasicQA, interpreter=MockInterpreter())


def test_pot_does_not_shutdown_caller_owned_interpreter():
    factory = MockInterpreterFactory()
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="SUBMIT({'answer': 2})")
    pot.generate_output = StaticPredictor(answer="2")
    interpreter = MockInterpreter(responses=[FinalOutput({"answer": "2"})])

    try:
        result = pot(interpreter, question="What is 1+1?")

        assert result.answer == "2"
        assert factory.instances == []
        assert interpreter.execute("print('still open')") == ""
    finally:
        interpreter.shutdown()


def test_pot_shuts_down_factory_interpreter_when_execution_raises():
    factory = MockInterpreterFactory(responses=[ValueError("unexpected interpreter failure")])
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="raise ValueError")

    with pytest.raises(ValueError, match="unexpected interpreter failure"):
        pot(question="What is 1+1?")

    assert len(factory.instances) == 1
    with pytest.raises(CodeInterpreterError, match="shutdown"):
        factory.instances[0].execute("print('closed')")


def test_pot_propagates_terminal_interpreter_failure_and_shuts_down():
    factory = MockInterpreterFactory(responses=[CodeInterpreterError("protocol corrupt")])
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="print('test')")

    with pytest.raises(CodeInterpreterError, match="protocol corrupt"):
        pot(question="What is 1+1?")

    assert len(factory.instances) == 1
    with pytest.raises(CodeInterpreterError, match="shutdown"):
        factory.instances[0].execute("print('closed')")


@pytest.mark.deno
def test_pot_code_generation_persistent_errors():
    max_iters = 3
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```",
            },
        ]
        * max_iters
    )
    dspy.configure(lm=lm)

    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with pytest.raises(RuntimeError, match=r"Max hops reached. Failed to run ProgramOfThought: ZeroDivisionError:"):
        pot(question="What is 1+1?")


def test_pot_code_parse_error():
    max_iters = 3
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\ninvalid=python=code\n```"},
        ]
        * max_iters
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with (
        patch("dspy.predict.program_of_thought.ProgramOfThought._execute_code") as mock_execute_code,
        pytest.raises(
            RuntimeError, match=r"Max hops reached. Failed to run ProgramOfThought: Error: Code format is not correct."
        ),
    ):
        pot(question="What is 1+1?")
    mock_execute_code.assert_not_called()


def test_pot_parse_code_appends_echo_for_single_line_assignment():
    """_parse_code appends a trailing bare-name echo of the last assigned variable so the
    interpreter's last-expression value is captured as output. This must also apply when
    the generated code is a single line, not just when it's the last of several lines --
    a bare `answer = 42` is a common, simple case an LM can legitimately generate."""
    pot = ProgramOfThought(BasicQA)

    code_block, error = pot._parse_code({"generated_code": "answer = 42"})

    assert error is None
    assert code_block == "answer = 42\nanswer"


def test_pot_parse_code_multiline_assignment_still_appends_echo():
    """Regression guard: the multi-line case (already covered before this fix) must keep
    working the same way."""
    pot = ProgramOfThought(BasicQA)

    code_block, error = pot._parse_code({"generated_code": "x = 1\nanswer = x + 41"})

    assert error is None
    assert code_block == "x = 1\nanswer = x + 41\nanswer"


@pytest.mark.parametrize(
    "generated_code,expected_block",
    [
        ("result = 5 == 3", "result = 5 == 3\nresult"),
        ("ok = 1 <= 2", "ok = 1 <= 2\nok"),
        ("flag = a != b", "flag = a != b\nflag"),
        ("is_equal = (x == y)", "is_equal = (x == y)\nis_equal"),
        ("result = (5 == 3)", "result = (5 == 3)\nresult"),
        ("x = a == b == c", "x = a == b == c\nx"),
        ("print(a == b)", "print(a == b)"),
    ],
)
def test_pot_parse_code_accepts_single_line_comparison(generated_code, expected_block):
    """Single-line code blocks whose extra `=` characters belong to comparison operators
    (==, !=, <=, >=) are single assignments and must be accepted by _parse_code, not
    rejected by the multi-assignment guard. The trailing-name echo must still be appended
    when the line is a `name = ...` assignment."""
    pot = ProgramOfThought(BasicQA)

    code_block, error = pot._parse_code({"generated_code": f"```python\n{generated_code}\n```"})

    assert error is None
    assert code_block == expected_block


@pytest.mark.parametrize(
    "generated_code",
    ["x += 1", "x <<= 1", "x //= 2", "x **= 2", "x += 1 if a == b else 0"],
)
def test_pot_parse_code_accepts_single_line_augmented_assignment(generated_code):
    """Augmented assignments (+=, -=, <<=, //=, **=, ...) are single assignments and must
    not be rejected; combining one with a comparison operator on the RHS must still pass.
    No trailing echo is appended because the line does not match the `name =` prefix."""
    pot = ProgramOfThought(BasicQA)

    code_block, error = pot._parse_code({"generated_code": f"```python\n{generated_code}\n```"})

    assert error is None
    assert code_block == generated_code


@pytest.mark.parametrize(
    "generated_code",
    ["a = b = 5", "a = 1; b = 2", "x += 1; y -= 2", "x <<= 1; y >>= 2", "invalid=python=code"],
)
def test_pot_parse_code_still_rejects_genuine_multi_assignment_single_line(generated_code):
    """Guard value preserved: genuine multi-assignment single-liners (chained `a = b = c`,
    semicolon-separated statements, multiple augmented assignments, or `k=v=w` shapes) are
    still rejected, because the output-capturing echo only handles a single trailing
    assignment."""
    pot = ProgramOfThought(BasicQA)

    _, error = pot._parse_code({"generated_code": f"```python\n{generated_code}\n```"})

    assert error == "Error: Code format is not correct."


def test_pot_executes_single_line_comparison_code_without_retry():
    """End-to-end regression: a single-line code block containing a comparison operator
    (==, !=, <=, >=) must be dispatched to the interpreter and succeed on the first hop,
    not be rejected by _parse_code. Uses a MockInterpreter so no Deno runtime is required.
    Before the fix, _parse_code returned 'Error: Code format is not correct.', wasting
    retries and raising 'Max hops reached' if the LM held the same shape across hops."""
    factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "False"})])
    pot = ProgramOfThought(BasicQA, max_iters=3, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="```python\nresult = 5 == 3\n```")
    pot.generate_output = StaticPredictor(answer="False")

    result = pot(question="Is 5 equal to 3?")

    assert result.answer == "False"
    assert len(factory.instances) == 1
    assert factory.instances[0].call_count == 1
    assert factory.instances[0].call_history[0][0] == "result = 5 == 3\nresult"


@pytest.mark.deno
def test_old_style_pot_single_line_comparison(pooled_interpreter):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 5 == 3\n```"},
            {"reasoning": "Reason_B", "answer": "False"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(pooled_interpreter, question="Is 5 equal to 3?")
    assert res.answer == "False"
