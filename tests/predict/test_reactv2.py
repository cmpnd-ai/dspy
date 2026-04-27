import dspy
from dspy.adapters.types.history import (
    ActionEvent,
    History,
    InputEvent,
    Observation,
    OutputEvent,
    truncate_oldest_actions,
)
from dspy.adapters.types.tool import Tool, ToolCalls, _sanitize_tool_name
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
    # input + 2 actions + final = 4 events
    assert len(result.history.messages) == 4
    assert isinstance(result.history.messages[0], InputEvent)
    assert result.history.messages[0].event == "input"
    assert isinstance(result.history.messages[-1], OutputEvent)
    assert result.history.messages[-1].event == "output"


def test_max_iters_forced_submit():
    """VAL-CORE-004: model submits on final iteration within the loop."""
    lm = DummyLM([
        {"next_thought": "Adding.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        {"next_thought": "Adding again.", "tool_calls": [{"name": "add", "args": {"a": 3, "b": 4}}]},
        # Submit on the 3rd (final) iteration within the loop:
        {"next_thought": "Submitting.", "tool_calls": [{"name": "submit", "args": {"answer": "10"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    result = react(question="Add stuff", max_iters=3)
    assert result.answer == "10"


def test_per_call_max_iters():
    """VAL-CORE-007: agent(question=..., max_iters=2) overrides instance default."""
    lm = DummyLM([
        {"next_thought": "Adding.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
        # Submit on the 2nd (final) iteration within the loop:
        {"next_thought": "Submitting.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
    ])
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()], max_iters=20)
    result = react(question="1+2", max_iters=2)
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
    actions = [m for m in result.history.messages if isinstance(m, ActionEvent)]
    assert any("Unknown tool" in str(m.observations) for m in actions)


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
    actions = [m for m in result.history.messages if isinstance(m, ActionEvent)]
    assert any("Execution error" in str(m.observations) for m in actions)


def test_reactv2_exported_from_dspy():
    """ReActV2 exported from dspy."""
    assert hasattr(dspy, "ReActV2")
    assert dspy.ReActV2 is ReActV2


# --- History semantic events tests (VAL-HIST-*) ---

def test_history_events_input_action_final():
    """VAL-HIST-001: append methods create input/action/final events."""
    h = History(messages=[])
    h.append_input({"question": "hi"})
    h.append_action(thought="thinking", tool_calls=None, observations=[Observation(value="ok", is_error=False)])
    h.append_output({"answer": "bye"})
    assert [m.event for m in h.messages] == ["input", "action", "output"]
    assert isinstance(h.messages[0], InputEvent)
    assert h.messages[0].inputs["question"] == "hi"
    assert isinstance(h.messages[1], ActionEvent)
    assert h.messages[1].thought == "thinking"
    assert h.messages[1].observations == [Observation(value="ok", is_error=False)]
    assert isinstance(h.messages[2], OutputEvent)
    assert h.messages[2].outputs["answer"] == "bye"


def test_has_open_episode():
    """VAL-HIST-002: has_open_episode tracks state correctly."""
    h = History(messages=[])
    assert not h.has_open_episode()
    h.append_input({"q": "1"})
    assert h.has_open_episode()
    h.append_action(thought="t", tool_calls=None, observations=[])
    assert h.has_open_episode()
    h.append_output({"a": "1"})
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
    requests = [m for m in r2.history.messages if isinstance(m, InputEvent)]
    assert len(requests) == 2


# --- Compaction tests (VAL-COMPACT-*) ---

def test_truncate_oldest_actions():
    """VAL-COMPACT-001: truncation preserves input event + most recent N actions."""
    h = History(messages=[
        InputEvent(inputs={"q": "x"}),
        *[ActionEvent(thought=str(i)) for i in range(10)],
    ])
    truncate_oldest_actions(h, max_tokens=0, keep_n=3)
    actions = [m for m in h.messages if isinstance(m, ActionEvent)]
    assert len(actions) == 3
    assert [a.thought for a in actions] == ["7", "8", "9"]
    assert isinstance(h.messages[0], InputEvent)


def test_compaction_is_callers_responsibility():
    """VAL-COMPACT-002: compact_if_needed() is NOT called inside forward(); callers manage compaction."""
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
    # compact_if_needed is no longer called inside forward()
    assert len(calls) == 0


# --- Native FC + format tests (VAL-FMT-*) ---

def test_native_fc_prompt_format():
    """VAL-FMT-002: Native path has reasoning guidance, no [[ ## completed ## ]]."""
    adapter = dspy.ChatAdapter(use_native_function_calling=True)
    sig = (
        dspy.Signature({}, "Do the task.")
        .append("question", dspy.InputField(), type_=str)
        .append("tools", dspy.InputField(), type_=list[dspy.Tool])
        .append("next_thought", dspy.OutputField(), type_=str)
        .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
    )
    # Simulate _call_preprocess: remove tool_calls and tools, tag native FC
    processed = sig.delete("tool_calls").delete("tools")
    processed.__dspy_native_fc__ = True
    messages = adapter.format(processed, [], {"question": "hi"})
    system_msg = messages[0]["content"]
    assert "[[ ## completed ## ]]" not in system_msg
    assert "step-by-step" in system_msg.lower() or "reasoning" in system_msg.lower() or "tool" in system_msg.lower()


def test_non_native_prompt_format_unchanged():
    """VAL-FMT-001: Non-native path still has structured markers."""
    adapter = dspy.ChatAdapter()
    sig = (
        dspy.Signature({}, "Do the task.")
        .append("question", dspy.InputField(), type_=str)
        .append("next_thought", dspy.OutputField(), type_=str)
        .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
    )
    messages = adapter.format(sig, [], {"question": "hi"})
    system_msg = messages[0]["content"]
    assert "[[ ## completed ## ]]" in system_msg
    assert "[[ ## next_thought ## ]]" in system_msg


def test_toolcalls_normalizes_openai_format():
    """VAL-FMT-004: ToolCalls normalizes OpenAI {type:'function', function:{name, arguments}} format."""
    tc = ToolCalls(tool_calls=[
        {"type": "function", "function": {"name": "search", "arguments": {"query": "hello"}}},
        {"type": "function", "function": {"name": "submit", "arguments": {"answer": "42"}}},
    ])
    assert len(tc.tool_calls) == 2
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[0].args == {"query": "hello"}
    assert tc.tool_calls[1].name == "submit"


def test_tool_name_sanitization():
    """Tool names sanitized to match OpenAI ^[a-zA-Z0-9_-]+$ pattern."""
    assert _sanitize_tool_name("my.tool") == "my_tool"
    assert _sanitize_tool_name("tool name!") == "tool_name_"
    assert _sanitize_tool_name("valid-name_123") == "valid-name_123"
    # Test via Tool constructor
    def my_weird_fn(x: str) -> str:
        """A tool."""
        return x
    tool = Tool(my_weird_fn, name="weird.tool.name")
    assert tool.name == "weird_tool_name"


def test_supports_fc_provider_fallback():
    """gpt-5-nano reports supports_fc=True via provider fallback."""
    lm = dspy.LM("openai/gpt-5-nano", cache=False)
    assert lm.supports_function_calling is True


def test_gepa_compile_with_reactv2():
    """VAL-OPTIM-001: GEPA.compile() on a ReActV2 module completes without error."""
    from dspy.teleprompt.gepa.gepa import GEPA
    lm = DummyLM([
        {"next_thought": "Do it.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
    ] * 20)
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[_make_add_tool()])
    metric = lambda ex, pred, *a, **kw: float(getattr(pred, "answer", None) == ex.answer) if hasattr(pred, "answer") else 0.0
    trainset = [dspy.Example(question="1+2", answer="3").with_inputs("question")]
    gepa = GEPA(metric=metric, max_metric_calls=2, reflection_lm=lm)
    result = gepa.compile(react, trainset=trainset)
    assert isinstance(result, ReActV2)
    assert "add" in result.react.signature.instructions
