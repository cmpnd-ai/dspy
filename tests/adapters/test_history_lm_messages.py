import json

import dspy
from dspy.adapters.types.history import History, Observation
from dspy.adapters.types.tool import Tool, ToolCalls


def add(a: int, b: int) -> int:
    return a + b


class NativeToolLM:
    model = "openai/gpt-5-nano"
    supported_params = frozenset()
    supports_function_calling = True
    supports_reasoning = False
    supports_response_schema = False

    def __call__(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [{"text": "[[ ## next_thought ## ]]\nDone.\n\n[[ ## completed ## ]]"}]


def test_history_renders_native_tool_messages_through_lm_message_path():
    adapter = dspy.ChatAdapter(use_native_function_calling=True)
    signature = (
        dspy.Signature({}, "Do the task.")
        .append("question", dspy.InputField(), type_=str)
        .append("history", dspy.InputField(), type_=dspy.History)
        .append("tools", dspy.InputField(), type_=list[dspy.Tool])
        .append("next_thought", dspy.OutputField(), type_=str)
        .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
    )
    history = History(frames=[])
    history.append_inputs({"question": "What is 1+2?"})
    history.append_outputs(
        {
            "next_thought": "I should add the two numbers.",
            "tool_calls": ToolCalls.from_dict_list(
                [{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_add"}]
            ),
        },
        observations=[Observation(value=3, source="tool", call_id="call_add", name="add")],
    )

    lm = NativeToolLM()
    adapter(lm, {}, signature, [], {"question": "What is 1+2?", "history": history, "tools": [Tool(add)]})

    assistant_message = next(message for message in lm.messages if message["role"] == "assistant")
    tool_message = next(message for message in lm.messages if message["role"] == "tool")
    final_user_message = lm.messages[-1]

    assert assistant_message["content"] == "I should add the two numbers."
    assert assistant_message["tool_calls"][0]["id"] == "call_add"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "add"
    assert json.loads(assistant_message["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "b": 2}
    assert tool_message == {"role": "tool", "content": "3", "tool_call_id": "call_add", "name": "add"}
    assert "What is 1+2?" not in final_user_message["content"]
    assert "Respond with the corresponding output fields" in final_user_message["content"]
