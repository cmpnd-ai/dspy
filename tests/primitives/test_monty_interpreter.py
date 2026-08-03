from __future__ import annotations

import pytest

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreter, CodeInterpreterError, FinalOutput
from dspy.primitives.monty_interpreter import MontyInterpreter


def test_implements_code_interpreter() -> None:
    assert isinstance(MontyInterpreter(), CodeInterpreter)


def test_persistent_execution_and_submit() -> None:
    with MontyInterpreter() as interpreter:
        interpreter.execute("value = 40")
        assert interpreter.execute("value + 2") == 42
        interpreter.output_fields = [{"name": "answer"}]
        assert interpreter.execute("SUBMIT(value + 2)") == FinalOutput({"answer": 42})


def test_host_tool_accepts_positional_arguments() -> None:
    interpreter = MontyInterpreter(tools={"add": lambda a, b: a + b})
    try:
        assert interpreter.execute("add(20, 22)") == 42
    finally:
        interpreter.shutdown()


def test_guest_errors_are_recoverable() -> None:
    with MontyInterpreter() as interpreter:
        with pytest.raises(CodeExecutionError):
            interpreter.execute("missing_name")
        assert interpreter.execute("6 * 7") == 42


def test_shutdown_is_terminal() -> None:
    interpreter = MontyInterpreter()
    interpreter.start()
    interpreter.shutdown()
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1 + 1")
