from typing import ClassVar

import dspy
from dspy.core.types import LMOutput, LMResponse, LMTextPart
from dspy.utils.callback import BaseCallback


def search(query: str) -> str:
    return query


def add(a: int, b: int) -> int:
    return a + b


class NativeToolLM:
    model = "openai/gpt-5-nano"
    model_type = "chat"
    kwargs: ClassVar[dict] = {}
    supported_params = frozenset()
    supports_function_calling = True
    supports_reasoning = False
    supports_response_schema = False

    def __init__(self, output=None):
        self.output = output
        self.messages = None
        self.kwargs = None

    def __call__(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [self.output]


def test_prepare_request_state_copies_kwargs_and_extracts_tools():
    class ToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    adapter = dspy.Adapter(use_native_function_calling=True)
    lm_kwargs = {"temperature": 0.2}
    inputs = {"question": "Q?", "tools": [dspy.Tool(search)]}

    state = adapter._prepare_request_state(NativeToolLM(), lm_kwargs, ToolSignature, inputs)

    assert lm_kwargs == {"temperature": 0.2}
    assert inputs["tools"][0].name == "search"
    assert "tools" not in state.lm_kwargs
    assert state.tools[0].name == "search"
    assert "tools" not in state.render_signature.input_fields
    assert "tool_calls" not in state.render_signature.output_fields
    assert state.hidden_output_fields == ("tool_calls",)

def test_prepare_request_state_preserves_normal_signature_and_copies_data():
    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.Adapter()
    lm_kwargs = {"temperature": 0.2, "metadata": {"trace_id": "abc"}}
    inputs = {"question": "Q?"}

    state = adapter._prepare_request_state(NativeToolLM(), lm_kwargs, QASignature, inputs)

    assert state.source_signature is QASignature
    assert state.render_signature is QASignature
    assert state.inputs == inputs
    assert state.inputs is not inputs
    assert state.lm_kwargs == lm_kwargs
    assert state.lm_kwargs is not lm_kwargs
    assert state.tools == []
    assert state.hidden_output_fields == ()
    assert lm_kwargs == {"temperature": 0.2, "metadata": {"trace_id": "abc"}}
    assert inputs == {"question": "Q?"}


def test_render_request_normal_state_preserves_messages_and_kwargs():
    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    lm = NativeToolLM()
    state = adapter._prepare_request_state(
        lm,
        {"temperature": 0.7, "n": 2, "custom_option": "value"},
        QASignature,
        {"question": "Q2?"},
    )

    messages = adapter.format(state.render_signature, [{"question": "Q1?", "answer": "A1"}], state.inputs)
    request = adapter._render_request(lm, state, messages)

    assert request.model == lm.model
    assert request.tools == []
    assert request.config.temperature == 0.7
    assert request.config.n == 2
    assert request.config.extensions == {"custom_option": "value"}
    assert [message.role for message in request.messages] == ["system", "user", "assistant", "user"]
    assert "Q1?" in request.messages[1].text
    assert "A1" in request.messages[2].text
    assert "Q2?" in request.messages[3].text


def test_render_request_normal_state_expands_history_without_mutating_inputs():
    class HistorySignature(dspy.Signature):
        history: dspy.History = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    lm = NativeToolLM()
    history = dspy.History(messages=[{"question": "What is the capital of France?", "answer": "Paris"}])
    inputs = {"history": history, "question": "What country is it in?"}

    state = adapter._prepare_request_state(lm, {}, HistorySignature, inputs)
    messages = adapter.format(state.render_signature, [], state.inputs)
    request = adapter._render_request(lm, state, messages)

    assert inputs == {"history": history, "question": "What country is it in?"}
    assert [message.role for message in request.messages] == ["system", "user", "assistant", "user"]
    assert "What is the capital of France?" in request.messages[1].text
    assert "Paris" in request.messages[2].text
    assert "What country is it in?" in request.messages[3].text
    assert "[[ ## history ## ]]" not in request.messages[3].text
    assert "Paris" not in request.messages[3].text


def test_adapter_format_callbacks_receive_public_arguments():
    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class CaptureFormatCallback(BaseCallback):
        def __init__(self):
            self.inputs = None

        def on_adapter_format_start(self, call_id, instance, inputs):
            self.inputs = inputs

    callback = CaptureFormatCallback()
    lm = NativeToolLM("[[ ## answer ## ]]\nA\n\n[[ ## completed ## ]]")

    dspy.ChatAdapter(callbacks=[callback], use_json_adapter_fallback=False)(
        lm,
        {},
        QASignature,
        [{"question": "demo question", "answer": "demo answer"}],
        {"question": "Q?"},
    )

    assert set(callback.inputs) == {"signature", "demos", "inputs"}
    assert callback.inputs["signature"] is QASignature
    assert callback.inputs["demos"] == [{"question": "demo question", "answer": "demo answer"}]
    assert callback.inputs["inputs"] == {"question": "Q?"}


def test_call_path_uses_public_format_override():
    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class FormatOverrideAdapter(dspy.Adapter):
        def format(self, signature, demos, inputs):
            return [
                {"role": "system", "content": "custom system"},
                {"role": "user", "content": f"custom user: {inputs['question']}"},
            ]

        def parse(self, signature, completion):
            return {"answer": completion}

    lm = NativeToolLM("custom answer")

    result = FormatOverrideAdapter()(
        lm,
        {},
        QASignature,
        [{"question": "demo question", "answer": "demo answer"}],
        {"question": "Q?"},
    )

    assert result == [{"answer": "custom answer"}]
    assert lm.messages == [
        {"role": "system", "content": "custom system"},
        {"role": "user", "content": "custom user: Q?"},
    ]


def test_call_path_uses_public_format_demos_override():
    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class DemoOverrideAdapter(dspy.ChatAdapter):
        def format_demos(self, signature, demos):
            return [{"role": "user", "content": "custom demo message"}]

    lm = NativeToolLM("[[ ## answer ## ]]\nA\n\n[[ ## completed ## ]]")

    result = DemoOverrideAdapter(use_json_adapter_fallback=False)(
        lm,
        {},
        QASignature,
        [{"question": "demo question", "answer": "demo answer"}],
        {"question": "Q?"},
    )

    assert result == [{"answer": "A"}]
    assert any(message["content"] == "custom demo message" for message in lm.messages)
    assert not any("demo question" in message["content"] for message in lm.messages)


def test_parse_response_normal_state_parses_text_output_and_logprobs():
    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    state = adapter._prepare_request_state(NativeToolLM(), {}, QASignature, {"question": "Q?"})
    response = LMResponse(
        model="dummy",
        outputs=[
            LMOutput(
                parts=[LMTextPart(text="[[ ## answer ## ]]\nA\n\n[[ ## completed ## ]]")],
                logprobs={"tokens": ["A"]},
            )
        ],
    )

    assert adapter._parse_response(state, response) == [{"answer": "A", "logprobs": {"tokens": ["A"]}}]


def test_native_tool_response_can_combine_visible_text_and_tool_calls():
    class ToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    lm = NativeToolLM(
        {
            "text": "[[ ## answer ## ]]\nworking\n\n[[ ## completed ## ]]",
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                }
            ],
        }
    )

    result = dspy.ChatAdapter(use_native_function_calling=True)(
        lm,
        {},
        ToolSignature,
        [],
        {"question": "What is 1+2?", "tools": [dspy.Tool(add)]},
    )[0]

    assert result["answer"] == "working"
    assert result["tool_calls"].tool_calls[0].name == "add"
    assert result["tool_calls"].tool_calls[0].args == {"a": 1, "b": 2}
    assert "tools" in lm.kwargs
    assert lm.kwargs["tools"][0]["function"]["name"] == "add"


def test_native_tool_response_repairs_nonstandard_json_arguments():
    class ToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    lm = NativeToolLM(
        {
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{'a': 1, 'b': 2}"},
                }
            ],
        }
    )

    result = dspy.ChatAdapter(use_native_function_calling=True)(
        lm,
        {},
        ToolSignature,
        [],
        {"question": "What is 1+2?", "tools": [dspy.Tool(add)]},
    )[0]

    assert result["tool_calls"].tool_calls[0].args == {"a": 1, "b": 2}
