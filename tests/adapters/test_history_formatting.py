import json

import dspy
from dspy.adapters.types.history import History, Observation


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class AgentTurn(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    next_thought: str = dspy.OutputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


def _history_with_open_tool_turn() -> History:
    history = History(frames=[])
    history.append_inputs({"question": "What is 1+2?"})
    history.append_outputs(
        {
            "next_thought": "I should add.",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "add", "args": {"a": 1, "b": 2}}]),
        },
        observations=[Observation(value=3, source="tool", name="add")],
    )
    return history


def test_chat_adapter_formats_open_history_without_duplicate_input():
    messages = dspy.ChatAdapter().format(
        AgentTurn,
        [],
        {"question": "What is 1+2?", "history": _history_with_open_tool_turn(), "tools": [dspy.Tool(add)]},
    )

    texts = [message.text or "" for message in messages]

    assert [message.role for message in messages] == ["system", "user", "assistant", "user", "user"]
    assert sum("What is 1+2?" in text for text in texts[1:]) == 1
    assert "[[ ## next_thought ## ]]\nI should add." in texts[2]
    assert "[[ ## tool_calls ## ]]" in texts[2]
    assert "Tool call 1 (`add`):\nObservation: 3" in texts[3]
    assert "Respond with the corresponding output fields" in texts[4]


def test_json_adapter_formats_open_history_without_duplicate_input():
    messages = dspy.JSONAdapter().format(
        AgentTurn,
        [],
        {"question": "What is 1+2?", "history": _history_with_open_tool_turn(), "tools": [dspy.Tool(add)]},
    )

    texts = [message.text or "" for message in messages]
    assistant = json.loads(texts[2])

    assert [message.role for message in messages] == ["system", "user", "assistant", "user", "user"]
    assert sum("What is 1+2?" in text for text in texts[1:]) == 1
    assert assistant["next_thought"] == "I should add."
    assert assistant["tool_calls"]["tool_calls"][0]["function"]["name"] == "add"
    assert "Tool call 1 (`add`):\nObservation: 3" in texts[3]
    assert "Respond with a JSON object" in texts[4]
