import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, get_args

from pydantic.fields import FieldInfo

from dspy.adapters._legacy_type_markers import (
    _expand_legacy_custom_type_markers_in_lm_message,
    _split_legacy_custom_type_text_to_parts,
)
from dspy.adapters._type_feature_handlers import (
    _CitationsTypeHandler,
    _ReasoningTypeHandler,
    _RenderedTypeOutput,
    _ToolTypeHandler,
)
from dspy.adapters._type_runtime import (
    _AdapterCallPlan,
    _CallContext,
    _merge_lm_config,
    _OutputParser,
    _TypeFeatureHandler,
    _TypeParseContext,
    _TypeRenderContext,
)
from dspy.adapters.types import History, Type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.adapters.utils import format_field_value, parse_value
from dspy.clients.base_lm import BaseLM, LMCapabilities
from dspy.clients.openai_format import (
    lm_response_from_legacy_outputs,
    message_to_openai_chat,
    parts_to_openai_content,
    to_openai_chat_request,
)
from dspy.core.types import LMConfig, LMMessage, LMOutput, LMPart, LMRequest, LMResponse, LMTextPart
from dspy.experimental import Citations
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]


@dataclass
class _RenderedAdapterRequest:
    request: LMRequest
    call_plan: _AdapterCallPlan
    context: _CallContext

    @property
    def signature(self) -> type[Signature]:
        return self.call_plan.render_signature


class _AdapterContextLM(BaseLM):
    """Minimal BaseLM used when rendering adapter messages without a real LM."""

    def __init__(self, *, use_native_tool_calls: bool = False):
        super().__init__(model="", model_type="chat", temperature=None, max_tokens=None)
        self._capabilities = LMCapabilities(function_calling=use_native_tool_calls)

    def get_capabilities(self) -> LMCapabilities:
        return self._capabilities


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
        allow_parallel_tool_calls: bool | None = None,
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
                (e.g., Anthropic's citation feature). Defaults to `[Citations, Reasoning]`.
            allow_parallel_tool_calls: Optional provider/tool-call policy. `False` enforces one call per turn for
                native and non-native tool calling; `None` preserves provider/model defaults.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.native_response_types = list(
            _DEFAULT_NATIVE_RESPONSE_TYPES if native_response_types is None else native_response_types
        )
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def build_call_context(self, lm: BaseLM, lm_kwargs: dict[str, Any] | None = None) -> _CallContext:
        return _CallContext(
            adapter=self,
            use_native_function_calling=self.use_native_function_calling,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
            native_response_types=tuple(self.native_response_types),
            lm=lm,
            lm_kwargs=dict(lm_kwargs or {}),
            lm_default_kwargs=dict(getattr(lm, "kwargs", {}) or {}),
        )

    def _default_call_context(self, *, use_native_tool_calls: bool = False) -> _CallContext:
        return _CallContext(
            adapter=self,
            use_native_function_calling=use_native_tool_calls,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
            native_response_types=tuple(self.native_response_types),
            lm=_AdapterContextLM(use_native_tool_calls=use_native_tool_calls),
        )

    def value_to_lm_parts(self, value: object, field_info: FieldInfo) -> list[LMPart]:
        rendered = format_field_value(field_info=field_info, value=value)
        return self.coerce_field_payload_to_lm_parts(rendered)

    def coerce_field_payload_to_lm_parts(self, rendered: object) -> list[LMPart]:
        if isinstance(rendered, str):
            if "<<CUSTOM-TYPE-START-IDENTIFIER>>" in rendered:
                return _split_legacy_custom_type_text_to_parts(rendered)
            return [LMTextPart(text=rendered)]
        if isinstance(rendered, list):
            return LMMessage.model_validate({"role": "user", "content": rendered}).parts
        if isinstance(rendered, dict):
            if "type" in rendered:
                return LMMessage.model_validate({"role": "user", "content": [rendered]}).parts
            return [LMTextPart(text=json.dumps(rendered, ensure_ascii=False))]
        return [LMTextPart(text=str(rendered))]

    def lm_parts_to_text(self, parts: list[LMPart]) -> str:
        texts: list[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if text is not None:
                texts.append(str(text))
            else:
                texts.append(str(part))
        return "".join(texts)

    def lm_parts_to_value(self, parts: list[LMPart], field_info: FieldInfo) -> object:
        text = self.lm_parts_to_text(parts)
        return parse_value(text, field_info.annotation)

    def wrap_input_field_parts(self, field_name: str, parts: list[LMPart]) -> list[LMPart]:
        raise NotImplementedError

    def wrap_output_field_parts(self, field_name: str, parts: list[LMPart]) -> list[LMPart]:
        raise NotImplementedError

    def parse_output_fields_to_parts(
        self,
        output: LMOutput,
        signature: type[Signature],
    ) -> dict[str, list[LMPart]]:
        parsed = self.parse(signature, output.text or "")
        return {
            field_name: self.value_to_lm_parts(parsed[field_name], field_info)
            for field_name, field_info in signature.output_fields.items()
            if field_name in parsed
        }

    def _parts_to_message_content(self, parts: list[LMPart]) -> str | list[dict[str, Any]]:
        return parts_to_openai_content(parts)

    def _make_type_render_context(
        self,
        *,
        field_name: str,
        field_info: FieldInfo,
        signature: type[Signature],
        values: dict[str, Any],
        role: str,
        context: _CallContext | None,
    ) -> _TypeRenderContext:
        return _TypeRenderContext(
            field_name=field_name,
            field_info=field_info,
            signature=signature,
            values=values,
            adapter=self,
            call_context=context or self._current_call_context(),
            role=role,
        )

    def _current_call_context(self) -> _CallContext:
        return getattr(self, "_call_context", None) or self._default_call_context()

    def _default_render_type_value(
        self,
        *,
        field_name: str,
        field_info: FieldInfo,
        signature: type[Signature],
        values: dict[str, Any],
        value: Any,
        role: str,
        context: _CallContext | None = None,
    ) -> list[LMPart] | None:
        annotation = self._type_annotation_class(field_info.annotation)
        if annotation is not None and not isinstance(value, annotation):
            try:
                value = annotation.model_validate(value)
            except Exception:
                pass

        if not isinstance(value, Type):
            return None

        render_ctx = self._make_type_render_context(
            field_name=field_name,
            field_info=field_info,
            signature=signature,
            values=values,
            role=role,
            context=context,
        )
        if role == "input":
            return value.default_render_input(render_ctx)
        return value.default_render_output(render_ctx)

    def _type_annotation_class(self, annotation: Any) -> type[Type] | None:
        try:
            if isinstance(annotation, type) and issubclass(annotation, Type):
                return annotation
        except TypeError:
            return None

        for arg in get_args(annotation):
            annotation_class = self._type_annotation_class(arg)
            if annotation_class is not None:
                return annotation_class
        return None

    def _merge_config_kwargs(self, kwargs: dict[str, Any], config: LMConfig | None) -> dict[str, Any]:
        if config is None:
            return kwargs
        base_config = LMConfig.from_kwargs(**kwargs)
        merged_config = _merge_lm_config(base_config, config)
        return merged_config.model_dump(exclude_none=True)

    def _merge_lm_config(self, left: LMConfig | None, right: LMConfig | None) -> LMConfig | None:
        return _merge_lm_config(left, right)

    def _render_request(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> _RenderedAdapterRequest:
        """Render the full signature into a normalized LM request."""
        context = self.build_call_context(lm, lm_kwargs)
        call_plan = _AdapterCallPlan.from_signature(signature, inputs, lm_kwargs)
        self._prepare_call_plan(call_plan, context)

        request_kwargs = self._prepare_request_kwargs(lm, call_plan.lm_kwargs, signature, call_plan.render_signature, context)
        tools = list(call_plan.tools)
        if "tools" in request_kwargs:
            tools.extend(request_kwargs.pop("tools") or [])
        request_kwargs = self._merge_config_kwargs(request_kwargs, call_plan.config)
        if not tools:
            request_kwargs.pop("tool_choice", None)

        messages = self._format_request_messages(call_plan, demos, context)
        request = LMRequest.from_call(
            model=lm.model,
            messages=self._coerce_lm_messages(messages),
            tools=tools,
            **request_kwargs,
        )
        return _RenderedAdapterRequest(request=request, call_plan=call_plan, context=context)

    def _type_feature_handlers(self) -> list[_TypeFeatureHandler]:
        return [_ToolTypeHandler(), _ReasoningTypeHandler(), _CitationsTypeHandler()]

    def _prepare_call_plan(self, call: _AdapterCallPlan, context: _CallContext) -> None:
        for field_name, field_info in call.source_signature.output_fields.items():
            annotation = self._type_annotation_class(field_info.annotation)
            if annotation is not None:
                call.output_parsers[field_name] = annotation.default_parse_output

        for handler in self._type_feature_handlers():
            handler.prepare(call, context)

    @staticmethod
    def _default_signature_instructions(signature: type[Signature]) -> str:
        inputs = ", ".join([f"`{field}`" for field in signature.input_fields])
        outputs = ", ".join([f"`{field}`" for field in signature.output_fields])
        if not outputs:
            return f"Given the fields {inputs}, follow the instructions."
        return f"Given the fields {inputs}, produce the fields {outputs}."

    def _format_request_messages(
        self,
        call_plan: _AdapterCallPlan,
        demos: list[dict[str, Any]],
        context: _CallContext,
    ) -> list[LMMessage | dict[str, Any]]:
        format_signature = inspect.signature(self.format)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in format_signature.parameters.values())
        kwargs: dict[str, Any] = {}
        if accepts_kwargs or "use_native_tool_calls" in format_signature.parameters:
            kwargs["use_native_tool_calls"] = bool(call_plan.tools)
        if accepts_kwargs or "context" in format_signature.parameters:
            kwargs["context"] = context
        if accepts_kwargs or "source_signature" in format_signature.parameters:
            kwargs["source_signature"] = call_plan.source_signature
        if accepts_kwargs or "prepared_messages" in format_signature.parameters:
            kwargs["prepared_messages"] = call_plan.messages
        sentinel = object()
        previous_context = getattr(self, "_call_context", sentinel)
        self._call_context = context
        try:
            messages = self.format(call_plan.render_signature, demos, call_plan.inputs, **kwargs)
        finally:
            if previous_context is sentinel:
                del self._call_context
            else:
                self._call_context = previous_context

        if not (accepts_kwargs or "prepared_messages" in format_signature.parameters):
            messages = [*messages, *call_plan.messages]
        return messages

    def _prepare_request_kwargs(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        source_signature: type[Signature],
        render_signature: type[Signature],
        context: _CallContext,
    ) -> dict[str, Any]:
        return dict(lm_kwargs)

    def _call_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        """Call current `BaseLM` through the normalized request/response boundary.

        TODO(language-models): When `BaseLM` is replaced by/updated to the
        normalized `BaseLM.forward(request: LMRequest) -> LMResponse` contract,
        remove this compatibility shim and let adapters call the normalized LM
        entry point directly. The OpenAI-shaped compatibility kwargs should live
        only inside concrete LM backends.
        """
        data = self._legacy_call_kwargs(request)
        outputs = lm(messages=data.pop("messages"), **data)
        return self._normalize_legacy_outputs(outputs, request)

    async def _acall_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        """Async variant of `_call_lm`.

        TODO(language-models): Same transitional boundary as `_call_lm()`; this
        should eventually call a normalized async LM method directly.
        """
        data = self._legacy_call_kwargs(request)
        outputs = await lm.acall(messages=data.pop("messages"), **data)
        return self._normalize_legacy_outputs(outputs, request)

    def _legacy_call_kwargs(self, request: LMRequest) -> dict[str, Any]:
        # TODO(language-models): Current `BaseLM` expects OpenAI/LiteLLM-shaped
        # chat kwargs. We intentionally use `dspy.clients.openai_format` here so
        # the conversion code lives in the future LM/client layer, not in
        # adapters. Remove this adapter helper once `BaseLM` accepts `LMRequest`.
        data = to_openai_chat_request(request)
        data.pop("model", None)
        # TODO(language-models): `cache` and `rollout_id` are DSPy BaseLM
        # execution controls, not provider request fields. The future
        # normalized LM base should own them before provider-format conversion.
        if request.config.cache is not None:
            if request.config.cache.enabled is not None:
                data["cache"] = request.config.cache.enabled
            if request.config.cache.rollout_id is not None:
                data["rollout_id"] = request.config.cache.rollout_id
        return data

    def _coerce_lm_messages(self, messages: list[LMMessage | dict[str, Any]]) -> list[LMMessage]:
        """Normalize subclass `format()` output before the LM boundary.

        TODO(adapters-normalized-rendering): Adapter `format()` methods still
        return OpenAI-chat-shaped dictionaries. This coercion is the bridge until
        adapters render `LMMessage` / `LMPart` directly.
        """
        return [
            _expand_legacy_custom_type_markers_in_lm_message(
                message if isinstance(message, LMMessage) else self._chat_dict_to_lm_message(message)
            )
            for message in messages
        ]

    def _chat_dict_to_lm_message(self, message: dict[str, Any]) -> LMMessage:
        try:
            return LMMessage(**message)
        except Exception:
            # TODO(legacy-custom-types): Unknown OpenAI content blocks are
            # temporarily preserved as `legacy_content_block` metadata on an
            # empty text part so `openai_format` can round-trip them back to
            # current BaseLM calls. Replace this with either explicit opaque
            # provider parts or remove it when marker-based custom type
            # serialization is retired.
            message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                sanitized = []
                supported = {"text", "image_url", "input_audio", "file", "document", "video"}
                for block in content:
                    if isinstance(block, dict) and block.get("type") in supported:
                        sanitized.append(block)
                    elif isinstance(block, dict):
                        sanitized.append({"type": "text", "text": "", "metadata": {"legacy_content_block": block}})
                    else:
                        sanitized.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
                message["content"] = sanitized
            return LMMessage(**message)

    def _normalize_legacy_outputs(self, outputs: list[dict[str, Any] | str | None], request: LMRequest) -> LMResponse:
        """Convert current `BaseLM` outputs into a normalized `LMResponse`.

        TODO(language-models): Current `BaseLM` returns `list[str | dict | None]`.
        Future LMs should return `LMResponse` directly, making this method a
        compatibility-only path for old/custom LMs.
        """
        return lm_response_from_legacy_outputs(outputs, request)

    def _parse_response(
        self,
        source_signature: type[Signature],
        call_plan: _AdapterCallPlan,
        response: LMResponse,
        context: _CallContext,
    ) -> list[dict[str, Any]]:
        """Parse a normalized LM response into dictionaries for the source signature."""
        values = []

        for output in response.outputs:
            if output.metadata.get("empty_legacy_outputs"):
                continue

            field_parts: dict[str, list[LMPart]] = {}
            has_text_output = bool(output.text and call_plan.render_signature.output_fields)
            if has_text_output:
                field_parts = self.parse_output_fields_to_parts(output, call_plan.render_signature)

            value: dict[str, Any] = {}
            for field_name, field_info in call_plan.render_signature.output_fields.items():
                if field_name in field_parts:
                    value[field_name] = self._parse_field_from_parts(
                        field_name,
                        field_info,
                        call_plan.render_signature,
                        field_parts[field_name],
                        output,
                        context,
                        call_plan.output_parsers.get(field_name),
                    )

            before_type_handler_count = len(value)
            for handler in self._type_feature_handlers():
                handler.parse(value, output, call_plan, context)
            parsed_by_type_handler = len(value) > before_type_handler_count

            for field_name in source_signature.output_fields:
                value.setdefault(field_name, None)

            if not (field_parts or parsed_by_type_handler):
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=source_signature,
                    lm_response=str(output.to_output_dict()),
                    message="The LM returned an empty or null response.",
                )

            if output.logprobs is not None:
                value["logprobs"] = output.logprobs

            self._validate_tool_call_parallel_policy(source_signature, value)
            values.append(value)

        return values

    def _parse_field_from_parts(
        self,
        field_name: str,
        field_info: FieldInfo,
        signature: type[Signature],
        parts: list[LMPart],
        output: LMOutput,
        context: _CallContext,
        output_parser: _OutputParser | None = None,
    ) -> Any:
        if output_parser is not None:
            return output_parser(
                _TypeParseContext(
                    field_name=field_name,
                    field_info=field_info,
                    signature=signature,
                    adapter=self,
                    call_context=context,
                    parts=parts,
                    lm_output=output,
                )
            )

        annotation = self._type_annotation_class(field_info.annotation)
        if annotation is not None:
            return annotation.default_parse_output(
                _TypeParseContext(
                    field_name=field_name,
                    field_info=field_info,
                    signature=signature,
                    adapter=self,
                    call_context=context,
                    parts=parts,
                    lm_output=output,
                )
            )
        return self.lm_parts_to_value(parts, field_info)

    def _validate_tool_call_parallel_policy(self, signature: type[Signature], value: dict[str, Any]) -> None:
        if self.allow_parallel_tool_calls is not False:
            return

        tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
        if not tool_call_output_field_name or tool_call_output_field_name not in value:
            return

        tool_calls = value[tool_call_output_field_name]
        if tool_calls is None:
            return
        if not isinstance(tool_calls, ToolCalls):
            tool_calls = ToolCalls.model_validate(tool_calls)
            value[tool_call_output_field_name] = tool_calls
        tool_calls.validate_max_items(1)

    def __call__(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute the adapter pipeline: format inputs, call LM, and parse outputs.

        Args:
            lm: The Language Model instance to use for generation. Must be an instance of `dspy.BaseLM`.
            lm_kwargs: Additional keyword arguments to pass to the LM call (e.g., temperature, max_tokens). These are
                passed directly to the LM.
            signature: The DSPy signature associated with this LM call.
            demos: List of few-shot examples to include in the prompt. Each dictionary should contain keys matching the
                signature's input and output field names. Examples are formatted as user/assistant message pairs.
            inputs: The current input values for this call. Keys must match the signature's input field names.

        Returns:
            List of dictionaries representing parsed LM responses. Each dictionary contains keys matching the
            signature's output field names. For multiple generations (n > 1), returns multiple dictionaries.
        """
        rendered = self._render_request(lm, lm_kwargs, signature, demos, inputs)
        response = self._call_lm(lm, rendered.request)
        return self._parse_response(signature, rendered.call_plan, response, rendered.context)

    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rendered = self._render_request(lm, lm_kwargs, signature, demos, inputs)
        response = await self._acall_lm(lm, rendered.request)
        return self._parse_response(signature, rendered.call_plan, response, rendered.context)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        *,
        use_native_tool_calls: bool = False,
        context: _CallContext | None = None,
        source_signature: type[Signature] | None = None,
        prepared_messages: list[LMMessage] | None = None,
    ) -> list[dict[str, Any]]:
        """Format the input messages for the LM call.

        This method converts the DSPy structured input along with few-shot examples and conversation history into
        multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
        the formatting of the input messages.

        In general we recommend the messages to have the following structure:
        ```
        [
            {"role": "system", "content": system_message},
            # Begin few-shot examples
            {"role": "user", "content": few_shot_example_1_input},
            {"role": "assistant", "content": few_shot_example_1_output},
            {"role": "user", "content": few_shot_example_2_input},
            {"role": "assistant", "content": few_shot_example_2_output},
            ...
            # End few-shot examples
            # Begin conversation history
            {"role": "user", "content": conversation_history_1_input},
            {"role": "assistant", "content": conversation_history_1_output},
            {"role": "user", "content": conversation_history_2_input},
            {"role": "assistant", "content": conversation_history_2_output},
            ...
            # End conversation history
            {"role": "user", "content": current_input},
        ]

        And system message should contain the field description, field structure, and task description.
        ```


        Args:
            signature: The DSPy signature for which to format the input messages.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        return [
            message_to_openai_chat(message)
            for message in self.render_messages(
                signature,
                demos,
                inputs,
                use_native_tool_calls=use_native_tool_calls,
                context=context,
                source_signature=source_signature,
                prepared_messages=prepared_messages,
            )
        ]

    def render_messages(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        *,
        use_native_tool_calls: bool = False,
        context: _CallContext | None = None,
        source_signature: type[Signature] | None = None,
        prepared_messages: list[LMMessage] | None = None,
    ) -> list[LMMessage]:
        """Render adapter messages as normalized LM messages."""
        context = context or self._default_call_context(use_native_tool_calls=use_native_tool_calls)
        source_signature = source_signature or signature
        inputs_copy = dict(inputs)

        render_signature = signature
        conversation_history: list[LMMessage] = []
        history_field_name = self._get_history_field_name(source_signature)
        if history_field_name:
            if history_field_name in render_signature.fields:
                render_signature = render_signature.delete(history_field_name)
                if source_signature.instructions == self._default_signature_instructions(source_signature):
                    instruction_signature = source_signature.delete(history_field_name)
                    render_signature = render_signature.with_instructions(
                        self._default_signature_instructions(instruction_signature)
                    )
            history_obj = inputs_copy.pop(history_field_name, None)
            if history_obj is not None:
                conversation_history = self.format_history(
                    history_obj,
                    source_signature.delete(history_field_name),
                    context=context,
                )

        messages: list[LMMessage | dict[str, Any]] = [
            {"role": "system", "content": self.format_system_message(render_signature)}
        ]
        messages.extend(
            self.format_demos(
                source_signature.delete(history_field_name) if history_field_name else source_signature,
                demos,
                context=context,
            )
        )
        messages.extend(conversation_history)
        messages.extend(prepared_messages or [])
        messages.extend(
            self._format_input_messages(
                render_signature,
                inputs_copy,
                main_request=True,
                context=context,
            )
        )

        return self._coerce_lm_messages(messages)

    def format_history(
        self,
        history: History,
        signature: type[Signature],
        *,
        use_native_tool_calls: bool = False,
        context: _CallContext | None = None,
    ) -> list[LMMessage]:
        """Render stored field history using this adapter's field format."""
        context = context or self._default_call_context(use_native_tool_calls=use_native_tool_calls)
        messages: list[LMMessage] = []
        for message_idx, entry in enumerate(history.messages):
            if not isinstance(entry, dict):
                continue

            input_values = {key: value for key, value in entry.items() if key in signature.input_fields}
            tool_result_values = {
                key: value
                for key, value in input_values.items()
                if self._coerce_tool_call_results_value(value, signature.input_fields.get(key)) is not None
            }
            output_values = {key: value for key, value in entry.items() if key in signature.output_fields}
            unknown_values = {
                key: value for key, value in entry.items() if key not in input_values and key not in output_values
            }

            turn_call = _AdapterCallPlan.from_signature(signature, input_values, context.lm_kwargs)
            self._prepare_call_plan(turn_call, context)

            regular_input_values = {
                key: turn_call.inputs[key]
                for key in turn_call.render_signature.input_fields
                if key in turn_call.inputs and key not in tool_result_values
            }
            if regular_input_values:
                messages.extend(
                    self._format_input_messages(
                        turn_call.render_signature,
                        regular_input_values,
                        main_request=False,
                        context=context,
                    )
                )

            assistant_values = dict(output_values)
            if not turn_call.messages:
                assistant_values.update(unknown_values)

            if assistant_values:
                messages.extend(
                    self._format_output_messages_for_call(
                        turn_call,
                        assistant_values,
                        message_idx,
                        context=context,
                    )
                )

            messages.extend(turn_call.messages)

            fallback_tool_result_values = {
                key: value
                for key, value in tool_result_values.items()
                if key in turn_call.render_signature.input_fields and not turn_call.messages
            }
            if fallback_tool_result_values:
                messages.extend(
                    self._format_input_messages(
                        turn_call.render_signature,
                        fallback_tool_result_values,
                        main_request=False,
                        context=context,
                    )
                )

            if (turn_call.messages or fallback_tool_result_values) and unknown_values:
                messages.extend(
                    self._format_output_messages_for_call(
                        turn_call,
                        unknown_values,
                        message_idx,
                        context=context,
                    )
                )

        return messages

    def _format_input_messages(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        *,
        main_request: bool,
        context: _CallContext | None = None,
        prefix: str = "",
        suffix: str = "",
    ) -> list[LMMessage]:
        context = context or self._current_call_context()
        regular_inputs = self._drop_absent_optional_inputs(signature, dict(inputs))
        if self._should_render_input_as_parts(signature, regular_inputs, context):
            parts: list[LMPart] = []
            if prefix:
                parts.append(LMTextPart(text=prefix))
            field_seen = False
            for key, field_info in signature.input_fields.items():
                if key not in regular_inputs:
                    continue
                if field_seen:
                    parts.append(LMTextPart(text="\n\n"))
                value_parts = self._default_render_type_value(
                    field_name=key,
                    field_info=field_info,
                    signature=signature,
                    values=regular_inputs,
                    value=regular_inputs[key],
                    role="input",
                    context=context,
                )
                if value_parts is None:
                    value_parts = self.value_to_lm_parts(regular_inputs[key], field_info)
                parts.extend(self.wrap_input_field_parts(key, value_parts))
                field_seen = True
            if main_request:
                output_requirements_fn = getattr(self, "user_message_output_requirements", lambda _signature: None)
                output_requirements = output_requirements_fn(signature)
                if output_requirements is not None:
                    if parts:
                        parts.append(LMTextPart(text="\n\n"))
                    parts.append(LMTextPart(text=output_requirements))
            if suffix:
                parts.append(LMTextPart(text=suffix))
            parts = self._merge_adjacent_text_parts(parts)
            return [LMMessage(role="user", parts=parts)] if parts else []

        content = self.format_user_message_content(
            signature,
            regular_inputs,
            prefix=prefix,
            suffix=suffix,
            main_request=main_request,
        )
        if self._has_content(content):
            return [self._content_message("user", content)]
        return []

    def _format_output_messages_for_call(
        self,
        call: _AdapterCallPlan,
        outputs: dict[str, Any],
        message_idx: int,
        *,
        context: _CallContext | None = None,
        missing_field_message: str | None = "Not supplied for this conversation history message. ",
    ) -> list[LMMessage]:
        context = context or self._current_call_context()
        regular_outputs = {}
        native_parts: list[LMPart] = []
        native_messages: list[LMMessage] = []

        for key, value in outputs.items():
            if key in call.render_signature.output_fields:
                regular_outputs[key] = value
                continue

            rendered_type = self._format_type_feature_output(key, value, call, context)
            if rendered_type is not None and rendered_type.consumed:
                native_parts.extend(rendered_type.parts)
                native_messages.extend(rendered_type.messages)
                for message in rendered_type.messages:
                    if message.role == "assistant":
                        native_parts.extend(message.parts)
                continue
            regular_outputs[key] = value

        if native_parts:
            content = self._format_native_output_content(
                call.render_signature,
                regular_outputs,
                missing_field_message=missing_field_message,
            )
            parts: list[LMPart] = []
            if content:
                parts.append(LMTextPart(text=content))
            parts.extend(native_parts)
            return [LMMessage(role="assistant", parts=parts)]

        if native_messages:
            return native_messages

        return [
            self._content_message(
                "assistant",
                self._format_output_content(
                    call.render_signature,
                    regular_outputs,
                    missing_field_message=missing_field_message,
                ),
            )
        ]

    def _format_type_feature_output(
        self,
        field_name: str,
        value: object,
        call: _AdapterCallPlan,
        context: _CallContext,
    ) -> _RenderedTypeOutput | None:
        for handler in self._type_feature_handlers():
            formatter = getattr(handler, "format_output", None)
            if formatter is None:
                continue
            rendered_type = formatter(field_name, value, call, context)
            if rendered_type is not None:
                return rendered_type
        return None

    def _format_output_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        *,
        missing_field_message: str | None,
    ) -> str:
        signature_outputs = {key: value for key, value in outputs.items() if key in signature.output_fields}
        unknown_outputs = {key: value for key, value in outputs.items() if key not in signature.output_fields}
        if signature_outputs and not unknown_outputs:
            return self.format_assistant_message_content(
                signature,
                signature_outputs,
                missing_field_message=missing_field_message,
            )

        sections = []
        if signature_outputs:
            sections.append(
                self.format_assistant_message_content(
                    signature,
                    signature_outputs,
                    missing_field_message=missing_field_message,
                ).strip()
            )

        for key, value in unknown_outputs.items():
            sections.append(f"[[ ## {key} ## ]]\n{self._format_value(value)}")

        sections.append("[[ ## completed ## ]]")
        return "\n\n".join(section for section in sections if section)

    def _format_native_output_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        *,
        missing_field_message: str | None,
    ) -> str | None:
        if not outputs:
            return None
        if len(outputs) == 1:
            return str(next(iter(outputs.values())))
        return self._format_output_content(signature, outputs, missing_field_message=missing_field_message)

    @staticmethod
    def _drop_absent_optional_inputs(signature: type[Signature], inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in inputs.items()
            if not (value is None and key in signature.input_fields and signature.input_fields[key].default is None)
        }

    def _should_render_input_as_parts(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        context: _CallContext,
    ) -> bool:
        for key, value in inputs.items():
            field_info = signature.input_fields.get(key)
            if field_info is None:
                continue
            parts = self._default_render_type_value(
                field_name=key,
                field_info=field_info,
                signature=signature,
                values=inputs,
                value=value,
                role="input",
                context=context,
            )
            if parts is not None and any(not isinstance(part, LMTextPart) for part in parts):
                return True
        return False

    @staticmethod
    def _merge_adjacent_text_parts(parts: list[LMPart]) -> list[LMPart]:
        merged: list[LMPart] = []
        for part in parts:
            if (
                isinstance(part, LMTextPart)
                and merged
                and isinstance(merged[-1], LMTextPart)
                and not part.metadata
                and not merged[-1].metadata
            ):
                merged[-1] = LMTextPart(text=merged[-1].text + part.text)
            else:
                merged.append(part)
        return merged

    @staticmethod
    def _coerce_tool_call_results_value(field_value: Any, field_info: Any = None) -> ToolCallResults | None:
        if field_value is None:
            return None
        if isinstance(field_value, ToolCallResults):
            return field_value

        annotation = getattr(field_info, "annotation", None)
        if annotation is ToolCallResults or ToolCallResults in get_args(annotation):
            return ToolCallResults.model_validate(field_value)
        return None

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return str(value)

    @staticmethod
    def _has_content(content: Any) -> bool:
        if isinstance(content, str):
            return bool(content.strip())
        return bool(content)

    @staticmethod
    def _content_message(role: str, content: Any) -> LMMessage:
        return LMMessage.model_validate({"role": role, "content": content})

    def format_system_message(self, signature: type[Signature]) -> str:
        """Format the system message for the LM call.


        Args:
            signature: The DSPy signature for which to format the system message.
        """
        return (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format the field description for the system message.

        This method formats the field description for the system message. It should return a string that contains
        the field description for the input fields and the output fields.

        Args:
            signature: The DSPy signature for which to format the field description.

        Returns:
            A string that contains the field description for the input fields and the output fields.
        """
        raise NotImplementedError

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the field structure for the system message.

        This method formats the field structure for the system message. It should return a string that dictates the
        format the input fields should be provided to the LM, and the format the output fields will be in the response.
        Refer to the ChatAdapter and JsonAdapter for an example.

        Args:
            signature: The DSPy signature for which to format the field structure.
        """
        raise NotImplementedError

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description for the system message.

        This method formats the task description for the system message. In most cases this is just a thin wrapper
        over `signature.instructions`.

        Args:
            signature: The DSPy signature of the DSpy module.

        Returns:
            A string that describes the task.
        """
        raise NotImplementedError

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message content.

        This method formats the user message content, which can be used in formatting few-shot examples, conversation
        history, and the current input.

        Args:
            signature: The DSPy signature for which to format the user message content.
            inputs: The input arguments to the DSPy module.
            prefix: A prefix to the user message content.
            suffix: A suffix to the user message content.

        Returns:
            A string that contains the user message content.
        """
        raise NotImplementedError

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format the assistant message content.

        This method formats the assistant message content, which can be used in formatting few-shot examples,
        conversation history.

        Args:
            signature: The DSPy signature for which to format the assistant message content.
            outputs: The output fields to be formatted.
            missing_field_message: A message to be used when a field is missing.

        Returns:
            A string that contains the assistant message content.
        """
        raise NotImplementedError

    def format_demos(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        *,
        use_native_tool_calls: bool = False,
        context: _CallContext | None = None,
    ) -> list[LMMessage | dict[str, Any]]:
        """Format the few-shot examples.

        This method formats the few-shot examples as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the few-shot examples.
            demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
                the signature.

        Returns:
            A list of multiturn messages.
        """
        context = context or self._default_call_context(use_native_tool_calls=use_native_tool_calls)
        active_fields = signature.fields
        messages = []
        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."

        for demo in demos:
            is_complete = all(k in demo and demo[k] is not None for k in active_fields)
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)
            if not (is_complete or (has_input and has_output)):
                continue

            demo_call = _AdapterCallPlan.from_signature(
                signature,
                {key: demo[key] for key in signature.input_fields if key in demo},
                context.lm_kwargs,
            )
            self._prepare_call_plan(demo_call, context)
            demo_inputs = {
                key: demo_call.inputs[key]
                for key in demo_call.render_signature.input_fields
                if key in demo_call.inputs
            }
            demo_outputs = {key: demo[key] for key in signature.output_fields if key in demo}
            messages.extend(
                self._format_input_messages(
                    demo_call.render_signature,
                    demo_inputs,
                    main_request=False,
                    context=context,
                    prefix="" if is_complete else incomplete_demo_prefix,
                )
            )
            messages.extend(
                self._format_output_messages_for_call(
                    demo_call,
                    demo_outputs,
                    len(messages),
                    context=context,
                    missing_field_message=(
                        "Not supplied for this conversation history message. "
                        if is_complete
                        else "Not supplied for this particular example. "
                    ),
                )
            )
            messages.extend(demo_call.messages)

        return messages

    def _get_history_field_name(self, signature: type[Signature]) -> str | None:
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    def _get_tool_call_output_field_name(self, signature: type[Signature]) -> str | None:
        return self._find_tool_call_output_field_name(signature)

    @classmethod
    def _find_tool_call_output_field_name(cls, signature: type[Signature]) -> str | None:
        for name, field in signature.output_fields.items():
            if cls._annotation_includes_static(field.annotation, ToolCalls):
                return name
        return None

    @classmethod
    def _annotation_includes_static(cls, annotation: Any, target: type) -> bool:
        if annotation is target:
            return True
        return any(cls._annotation_includes_static(arg, target) for arg in get_args(annotation))

    def force_tool_call_config(self, tool_name: str) -> dict[str, Any]:
        if not self.use_native_function_calling:
            return {}
        return {"tool_choice": {"mode": "required", "allowed": [tool_name]}}

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the conversation history.

        This method formats the conversation history and the current input as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the conversation history.
            history_field_name: The name of the history field in the signature.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages.
        """
        conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

        if conversation_history is None:
            return []

        # Remove the history field from the inputs
        del inputs[history_field_name]
        history = History(messages=conversation_history)
        return [message_to_openai_chat(message) for message in self.format_history(history, signature)]

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            signature: The DSPy signature for which to parse the LM output.
            completion: The LM output to be parsed.

        Returns:
            A dictionary of the output fields.
        """
        raise NotImplementedError
