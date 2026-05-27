import json
from typing import Any, get_args, get_origin

from dspy.adapters._legacy_type_markers import (
    _expand_legacy_custom_type_markers_in_chat_message,
    _expand_legacy_custom_type_markers_in_lm_message,
)
from dspy.adapters.types import History, Type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import serialize_for_json
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import lm_response_from_legacy_outputs, to_openai_chat_request
from dspy.core.types import LMRequest, LMResponse
from dspy.experimental import Citations
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

_TOOL_CALL_RESULTS_SIGNATURE = Signature({"tool_call_results": (ToolCallResults, InputField())})
_SUPPORTED_CONTENT_BLOCKS = {"text", "image_url", "input_audio", "file", "document", "video"}


class Adapter:
    """Base Adapter class.

    The Adapter serves as the interface layer between DSPy module/signature and Language Models (LMs). It handles the
    complete transformation pipeline from DSPy inputs to LM calls and back to structured outputs.

    Key responsibilities:
        - Transform user inputs and signatures into properly formatted LM prompts, which also instructs the LM to format
            the response in a specific format.
        - Parse LM outputs into dictionaries matching the signature's output fields.
        - Enable/disable native LM features (function calling, citations, etc.) based on configuration.
        - Handle conversation history, few-shot examples, and custom type processing.

    The adapter pattern allows DSPy to work with different LM interfaces while maintaining a consistent programming
    model for users.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        parallel_tool_calls: bool | None = None,
    ):
        """
        Args:
            callbacks: List of callback functions to execute during `format()` and `parse()` methods. Callbacks can be
                used for logging, monitoring, or custom processing. Defaults to None (empty list).
            use_native_function_calling: Whether to enable native function calling capabilities when the LM supports it.
                If True, the adapter will automatically configure function calling when input fields contain `dspy.Tool`
                or `list[dspy.Tool]` types. Defaults to False.
            native_response_types: List of output field types that should be handled by native LM features rather than
                adapter parsing. For example, `dspy.Citations` can be populated directly by citation APIs
                (e.g., Anthropic's citation feature). Defaults to `[Citations]`.
            parallel_tool_calls: Whether to request provider-side parallel tool-call generation when native function
                calling is active. If None, the adapter does not set the provider option. Defaults to None.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.parallel_tool_calls = parallel_tool_calls
        self.native_response_types = native_response_types or [Citations, Reasoning]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def __call__(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature,
        demos,
        inputs,
    ) -> list[dict[str, Any]]:
        processed, request = self._prepare_call(lm, lm_kwargs, signature, demos, inputs)
        return self._call_postprocess(processed, signature, _call_lm(lm, request), lm, lm_kwargs)

    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature,
        demos,
        inputs,
    ) -> list[dict[str, Any]]:
        processed, request = self._prepare_call(lm, lm_kwargs, signature, demos, inputs)
        return self._call_postprocess(processed, signature, await _acall_lm(lm, request), lm, lm_kwargs)

    def _prepare_call(self, lm, lm_kwargs, signature, demos, inputs) -> tuple[type[Signature], LMRequest]:
        signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        legacy_messages = self.format(signature, demos, inputs)
        messages = [_preserve_unknown_content_blocks(message) for message in legacy_messages]
        request = LMRequest.from_call(model=lm.model, messages=messages, **lm_kwargs)
        messages = [_expand_legacy_custom_type_markers_in_lm_message(message) for message in request.messages]
        return signature, request.model_copy(update={"messages": messages})

    def _call_preprocess(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs,
    ) -> type[Signature]:
        tool_input = _tool_input_field_name(signature)
        tool_output = self._get_tool_call_output_field_name(signature)

        if not self.use_native_function_calling:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                lm_kwargs.pop(key, None)
        elif tool_output:
            if tool_input is None:
                raise ValueError(
                    f"You provided an output field {tool_output} to receive the tool calls information, but did not "
                    "provide any tools as the input. Please provide a list of tools as the input by adding an input "
                    "field with type `list[dspy.Tool]`."
                )
            if lm.supports_function_calling:
                tools = inputs[tool_input]
                tools = tools if isinstance(tools, list) else [tools]
                lm_kwargs["tools"] = [tool.format_as_litellm_function_call() for tool in tools]
                if self.parallel_tool_calls is not None:
                    lm_kwargs.setdefault("tool_choice", "auto")
                    if lm_kwargs.get("parallel_tool_calls") is None:
                        lm_kwargs["parallel_tool_calls"] = self.parallel_tool_calls
                signature = signature.delete(tool_output).delete(tool_input)
        for name, field in signature.output_fields.items():
            if self._is_native_response_type(field.annotation):
                signature = field.annotation.adapt_to_native_lm_feature(signature, name, lm, lm_kwargs)
        return signature

    def _call_postprocess(
        self,
        processed_signature,
        original_signature,
        response: LMResponse,
        lm,
        lm_kwargs,
    ) -> list[dict[str, Any]]:
        if not isinstance(response, LMResponse):
            response = lm_response_from_legacy_outputs(response, LMRequest(model=getattr(lm, "model", ""), messages=[]))

        values, tool_output = [], self._get_tool_call_output_field_name(original_signature)

        for output in response.outputs:
            if output.metadata.get("empty_provider_outputs") or output.metadata.get("empty_legacy_outputs"):
                continue

            text, logprobs, tool_calls = output.text, output.logprobs, output.tool_calls
            if tool_calls and tool_output:
                value = self._parse_or_empty(processed_signature, text)
            elif text:
                value = self.parse(processed_signature, text)
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            for field_name in original_signature.output_fields:
                value.setdefault(field_name, None)

            if tool_calls and tool_output:
                value[tool_output] = _provider_tool_calls(tool_calls)

            for name, field in original_signature.output_fields.items():
                if (
                    self._is_native_response_type(field.annotation)
                    and (parsed := field.annotation.parse_lm_output(output)) is not None
                ):
                    value[name] = parsed

            if logprobs:
                value["logprobs"] = logprobs

            values.append(value)
        return values

    def _parse_or_empty(self, signature, text):
        try:
            return self.parse(signature, text) if text and signature.output_fields else {}
        except AdapterParseError:
            return {}

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the input messages for the LM call.

        This method converts DSPy structured input, few-shot examples, and conversation history into multiturn messages
        as expected by the LM. Custom adapters can override it to customize message formatting.
        """
        inputs = dict(inputs)
        history_name = next(
            (name for name, field in signature.input_fields.items() if field.annotation == History),
            None,
        )
        prompt_signature = signature.delete(history_name) if history_name else signature
        messages = [
            {"role": "system", "content": self.format_system_message(signature)},
            *self.format_demos(signature, demos),
        ]

        if history_name:
            messages.extend(self.format_conversation_history(prompt_signature, history_name, inputs))

        if content := self.format_user_message_content(prompt_signature, inputs, main_request=True):
            messages.append({"role": "user", "content": content})

        return [_expand_legacy_custom_type_markers_in_chat_message(message) for message in messages]

    def format_system_message(self, signature: type[Signature]) -> str:
        """Format the system message for the LM call."""
        return (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )

    def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format the few-shot examples."""
        messages = []
        incomplete_prefix = "This is an example of the task, though some input or output fields are not supplied."

        for demo in demos:
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if all(k in demo and demo[k] is not None for k in signature.fields):
                prefix, missing = "", "Not supplied for this conversation history message. "
            elif has_input and has_output:
                prefix, missing = incomplete_prefix, "Not supplied for this particular example. "
            else:
                continue

            messages.extend(
                [
                    {"role": "user", "content": self.format_user_message_content(signature, demo, prefix=prefix)},
                    {
                        "role": "assistant",
                        "content": self.format_assistant_message_content(
                            signature,
                            demo,
                            missing_field_message=missing,
                        ),
                    },
                ]
            )

        return messages

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the conversation history."""
        history = inputs[history_field_name].messages if history_field_name in inputs else None
        if history is None:
            return []
        messages = [message for item in history for message in self._history_messages(signature, item)]
        del inputs[history_field_name]
        return messages

    def _history_messages(self, signature, message) -> list[dict[str, Any]]:
        tool_field, tool_calls = _tool_calls_from_message(message)
        tool_results = (
            ToolCallResults.model_validate(tool_calls.tool_call_results)
            if tool_calls and tool_calls.tool_call_results is not None
            else None
        )

        messages = []
        if content := self.format_user_message_content(signature, message):
            messages.append({"role": "user", "content": content})

        if self.use_native_function_calling and tool_calls:
            return messages + self._native_tool_history_messages(signature, message, tool_calls, tool_results)

        assistant_values = message
        if tool_field and tool_results:
            assistant_values = dict(message, **{tool_field: tool_calls.model_copy(update={"tool_call_results": None})})

        if content := self.format_assistant_message_content(signature, assistant_values):
            messages.append({"role": "assistant", "content": content})

        if tool_results:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(
                        _TOOL_CALL_RESULTS_SIGNATURE,
                        {"tool_call_results": tool_results},
                    ),
                }
            )

        return messages

    def _native_tool_history_messages(self, signature, message, tool_calls, tool_results) -> list[dict[str, Any]]:
        content_signature = signature
        for name, field in signature.output_fields.items():
            if field.annotation == ToolCalls or message.get(name) is None:
                content_signature = content_signature.delete(name)
        content = ""
        if content_signature.output_fields:
            content = self.format_assistant_message_content(content_signature, message)

        ids = [tool_call.id for tool_call in tool_calls.tool_calls]
        result_ids = [result.call_id for result in tool_results.tool_call_results] if tool_results else []
        results = tool_results if tool_results and ids == result_ids and all(ids) else None

        messages = []
        if content or results:
            assistant_message = {"role": "assistant", "content": content or None}
            if results:
                assistant_message["tool_calls"] = [
                    _tool_call_as_openai_message_tool_call(tool_call) for tool_call in tool_calls.tool_calls
                ]
            messages.append(assistant_message)

        if results:
            messages.extend(_tool_result_as_openai_message(result) for result in results.tool_call_results)

        return messages

    def _get_tool_call_output_field_name(self, signature: type[Signature]) -> str | None:
        return next((name for name, field in signature.output_fields.items() if field.annotation == ToolCalls), None)

    def _is_native_response_type(self, annotation) -> bool:
        return (
            isinstance(annotation, type)
            and annotation in self.native_response_types
            and issubclass(annotation, Type)
        )


def _preserve_unknown_content_blocks(message):
    if not isinstance(message, dict) or not isinstance(content := message.get("content"), list):
        return message

    return {
        **message,
        "content": [
            block
            if isinstance(block, dict) and block.get("type") in _SUPPORTED_CONTENT_BLOCKS
            else _unknown_content_block(block)
            for block in content
        ],
    }


def _provider_tool_calls(tool_calls) -> ToolCalls:
    return ToolCalls.model_validate([_provider_tool_call_data(call) for call in tool_calls])


def _call_lm(lm, request: LMRequest) -> LMResponse:
    data = _legacy_call_kwargs(request)
    outputs = lm(messages=data.pop("messages"), **data)
    return lm_response_from_legacy_outputs(outputs, request)


async def _acall_lm(lm, request: LMRequest) -> LMResponse:
    data = _legacy_call_kwargs(request)
    outputs = await lm.acall(messages=data.pop("messages"), **data)
    return lm_response_from_legacy_outputs(outputs, request)


def _legacy_call_kwargs(request: LMRequest) -> dict[str, Any]:
    data = to_openai_chat_request(request)
    data.pop("model", None)
    if request.config.cache is not None:
        if request.config.cache.enabled is not None:
            data["cache"] = request.config.cache.enabled
        if request.config.cache.rollout_id is not None:
            data["rollout_id"] = request.config.cache.rollout_id
    return data


def _tool_calls_from_message(message: dict[str, Any]) -> tuple[str | None, ToolCalls | None]:
    return next(
        (
            (name, ToolCalls.model_validate(value))
            for name, value in message.items()
            if isinstance(value, ToolCalls) or (isinstance(value, dict) and "tool_calls" in value)
        ),
        (None, None),
    )


def _tool_input_field_name(signature: type[Signature]) -> str | None:
    return next(
        (name for name, field in signature.input_fields.items() if _is_tool_input_annotation(field.annotation)),
        None,
    )


def _is_tool_input_annotation(annotation) -> bool:
    return annotation == Tool or (get_origin(annotation) is list and get_args(annotation) == (Tool,))


def _tool_call_as_openai_message_tool_call(tool_call: ToolCalls.ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(serialize_for_json(tool_call.args), ensure_ascii=False),
        },
    }


def _tool_result_as_openai_message(result: ToolCallResults.ToolCallResult) -> dict[str, Any]:
    if isinstance(result.value, str):
        content = result.value
    else:
        content = json.dumps(serialize_for_json(result.value), ensure_ascii=False)

    return {"role": "tool", "tool_call_id": result.call_id, "name": result.name, "content": content}


def _unknown_content_block(block) -> dict[str, Any]:
    if isinstance(block, dict):
        return {"type": "text", "text": "", "metadata": {"legacy_content_block": block}}
    return {"type": "text", "text": json.dumps(block, ensure_ascii=False), "metadata": {}}


def _provider_tool_call_data(call):
    if hasattr(call, "model_dump"):
        return call.provider_data or call.model_dump()
    return call
