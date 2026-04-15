import dspy
from dspy.predict.reactv2 import ReActV2, _build_submit_tool
from dspy.utils.dummies import DummyLM


def _make_add_tool():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    return add


def test_submit_tool_returns_dict():
    """VAL-CORE-002: submit(answer='42') returns {'answer': '42'}."""
    sig = dspy.Signature("question -> answer")
    submit = _build_submit_tool(sig)
    result = submit(answer="42")
    assert result == {"answer": "42"}


def test_submit_tool_args_match_output_fields():
    """Submit tool args match signature output fields."""
    sig = dspy.Signature("question -> answer, confidence")
    submit = _build_submit_tool(sig)
    assert "answer" in submit.args
    assert "confidence" in submit.args
    result = submit(answer="42", confidence="high")
    assert result == {"answer": "42", "confidence": "high"}


def test_basic_forward_with_submit():
    """VAL-CORE-003: forward() terminates on submit, returns Prediction with history."""
    lm = DummyLM([
        {"next_thought": "I should add.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        {"next_thought": "I have the answer.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    result = react(question="What is 1+2?")
    assert result.answer == "3"
    assert hasattr(result, "history")
    assert len(result.history.messages) == 2


def test_max_iters_forced_submit():
    """VAL-CORE-004: max_iters exhausts triggers forced submit fallback."""
    lm = DummyLM([
        {"next_thought": "Adding.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        {"next_thought": "Adding again.", "tool_calls": [{"name": "add", "args": {"a": 3, "b": 4}}]},
        # Forced submit attempt:
        {"next_thought": "Submitting.", "tool_calls": [{"name": "submit", "args": {"answer": "10"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    result = react(question="Add stuff", max_iters=2)
    assert result.answer == "10"


def test_per_call_max_iters():
    """VAL-CORE-007: agent(question=..., max_iters=1) overrides instance default."""
    lm = DummyLM([
        {"next_thought": "Adding.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        # Forced submit:
        {"next_thought": "Submitting.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()], max_iters=20)
    result = react(question="1+2", max_iters=1)
    assert result.answer == "3"


def test_none_tool_calls_handled():
    """VAL-CORE-005: None tool_calls break loop gracefully."""
    lm = DummyLM([
        {"next_thought": "I dunno.", "tool_calls": []},
        # Forced submit - also returns empty so we get Prediction with just history
        {"next_thought": "Still nothing.", "tool_calls": []},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    result = react(question="What?")
    assert hasattr(result, "history")


def test_unknown_tool_returns_error_observation():
    """VAL-CORE-005: Unknown tool names return error observation, loop continues."""
    lm = DummyLM([
        {"next_thought": "Call fake.", "tool_calls": [{"name": "nonexistent", "args": {}}]},
        {"next_thought": "Now submit.", "tool_calls": [{"name": "submit", "args": {"answer": "ok"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    result = react(question="test")
    assert result.answer == "ok"
    # Check the history recorded the unknown tool error
    assert any("Unknown tool" in str(m.get("observations", "")) for m in result.history.messages)


def test_tool_execution_error_caught():
    """VAL-CORE-006: Tool exceptions caught as error observations, loop continues."""
    def failing_tool(x: str) -> str:
        """Always fails."""
        raise RuntimeError("boom")

    lm = DummyLM([
        {"next_thought": "Call it.", "tool_calls": [{"name": "failing_tool", "args": {"x": "hi"}}]},
        {"next_thought": "Submit anyway.", "tool_calls": [{"name": "submit", "args": {"answer": "recovered"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[failing_tool])
    result = react(question="test")
    assert result.answer == "recovered"
    assert any("Execution error" in str(m.get("observations", "")) for m in result.history.messages)


def test_reactv2_exported_from_dspy():
    """ReActV2 exported from dspy."""
    assert hasattr(dspy, "ReActV2")
    assert dspy.ReActV2 is ReActV2
