import dspy
from dspy.adapters.types.history import History, truncate_oldest_actions
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
    # REQUEST + 2 ACTIONs + FINAL = 4 events
    assert len(result.history.messages) == 4
    assert result.history.messages[0]["__dspy_history_event__"] == "REQUEST"
    assert result.history.messages[-1]["__dspy_history_event__"] == "FINAL"


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
    actions = [m for m in result.history.messages if m.get("__dspy_history_event__") == "ACTION"]
    assert any("Unknown tool" in str(m.get("observations", "")) for m in actions)


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
    actions = [m for m in result.history.messages if m.get("__dspy_history_event__") == "ACTION"]
    assert any("Execution error" in str(m.get("observations", "")) for m in actions)


def test_reactv2_exported_from_dspy():
    """ReActV2 exported from dspy."""
    assert hasattr(dspy, "ReActV2")
    assert dspy.ReActV2 is ReActV2


# --- History semantic events tests (VAL-HIST-*) ---

def test_history_events_request_action_final():
    """VAL-HIST-001: add_message creates REQUEST/ACTION/FINAL events."""
    h = History(messages=[])
    h.append_request({"question": "hi"})
    h.append_action(thought="thinking", tool_calls=None, observations=[("ok", False)])
    h.append_final({"answer": "bye"})
    assert [m["__dspy_history_event__"] for m in h.messages] == ["REQUEST", "ACTION", "FINAL"]
    assert h.messages[0]["question"] == "hi"
    assert h.messages[2]["answer"] == "bye"


def test_has_open_episode():
    """VAL-HIST-002: has_open_episode tracks state correctly."""
    h = History(messages=[])
    assert not h.has_open_episode()
    h.append_request({"q": "1"})
    assert h.has_open_episode()
    h.append_action(thought="t", tool_calls=None, observations=[])
    assert h.has_open_episode()
    h.append_final({"a": "1"})
    assert not h.has_open_episode()


def test_multi_turn_history_reuse():
    """VAL-HIST-003: History from forward #1 passed to forward #2."""
    lm = DummyLM([
        {"next_thought": "Add.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        {"next_thought": "Submit.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
        {"next_thought": "Add again.", "tool_calls": [{"name": "add", "args": {"a": 3, "b": 4}}]},
        {"next_thought": "Submit.", "tool_calls": [{"name": "submit", "args": {"answer": "7"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    r1 = react(question="1+2")
    r2 = react(question="3+4", history=r1.history)
    assert r2.answer == "7"
    requests = [m for m in r2.history.messages if m.get("__dspy_history_event__") == "REQUEST"]
    assert len(requests) == 2


# --- Compaction tests (VAL-COMPACT-*) ---

def test_truncate_oldest_actions():
    """VAL-COMPACT-001: truncation preserves REQUEST + most recent N actions."""
    h = History(messages=[
        {"__dspy_history_event__": "REQUEST", "q": "x"},
        *[{"__dspy_history_event__": "ACTION", "step": i} for i in range(10)],
    ])
    truncate_oldest_actions(h, max_tokens=0, keep_n=3)
    actions = [m for m in h.messages if m.get("__dspy_history_event__") == "ACTION"]
    assert len(actions) == 3
    assert [a["step"] for a in actions] == [7, 8, 9]
    assert h.messages[0]["__dspy_history_event__"] == "REQUEST"


def test_compaction_fires_in_forward_loop():
    """VAL-COMPACT-002: compact_if_needed() is called each iteration with custom fn."""
    calls = []
    def track_compact(history):
        calls.append(len(history.messages))
    lm = DummyLM([
        {"next_thought": "Go.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        {"next_thought": "Done.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    history = dspy.History(messages=[], compact_fn=track_compact)
    react(question="1+2", history=history)
    assert len(calls) == 2  # called each iteration
