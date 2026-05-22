import dspy
from dspy.adapters.types.history import HistoryFrame
from dspy.predict.reactv2 import ReActV2, ToolObservation, _build_submit_tool
from dspy.utils.dummies import DummyLM


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_submit_tool_returns_dict():
    signature = dspy.Signature("question -> answer")
    submit = _build_submit_tool(signature)

    assert submit(answer="42") == {"answer": "42"}


def test_basic_forward_with_submit_records_history_frames():
    lm = DummyLM(
        [
            {"next_thought": "I should add.", "tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]},
            {"next_thought": "I have the answer.", "tool_calls": [{"name": "submit", "args": {"answer": "3"}}]},
        ]
    )
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[add])

    result = react(question="What is 1+2?")

    assert result.answer == "3"
    assert len(result.history.frames) == 4
    assert result.history.frames[0].inputs == {"question": "What is 1+2?"}
    assert result.history.frames[1].outputs["next_thought"] == "I should add."
    assert result.history.frames[1].observations[0].value == 3
    assert result.history.frames[-1].outputs == {"answer": "3"}
    assert result.history.frames[-1].complete


def test_unknown_tool_returns_error_observation():
    lm = DummyLM(
        [
            {"next_thought": "Call fake.", "tool_calls": [{"name": "nonexistent", "args": {}}]},
            {"next_thought": "Now submit.", "tool_calls": [{"name": "submit", "args": {"answer": "ok"}}]},
        ]
    )
    dspy.configure(lm=lm)
    react = ReActV2("question -> answer", tools=[add])

    result = react(question="test")

    assert result.answer == "ok"
    observations = [obs for frame in result.history.frames if isinstance(frame, HistoryFrame) for obs in frame.observations]
    assert any(obs.is_error and "Unknown tool" in str(obs.value) for obs in observations)


def test_append_tool_turn_records_observation_ids():
    history = dspy.History(frames=[])
    tool_calls = dspy.ToolCalls.from_dict_list([{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_add"}])

    ReActV2._append_tool_turn(
        history,
        next_thought="add",
        tool_calls=tool_calls,
        observations=[ToolObservation(value=3)],
    )

    assert history.frames[0].observations[0].call_id == "call_add"
    assert history.frames[0].observations[0].name == "add"
