import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

import pydantic
from litellm import ContextWindowExceededError
from pydantic import TypeAdapter

import dspy
from dspy.adapters.types import History, HistoryCompaction
from dspy.adapters.types.tool import Tool, ToolCalls, _resolve_json_schema_reference
from dspy.adapters.utils import parse_value, serialize_for_json
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.clients.lm import LM
    from dspy.signatures.signature import Signature


class ReAct(Module):
    _SUBMIT_TOOL_NAME = "submit"

    def __init__(
        self,
        signature: type["Signature"],
        tools: list[Callable],
        max_iters: int = 20,
        parallel_tool_calls: bool = True,
    ):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        This implementation runs over a raw chat transcript, uses tool-call outputs natively when the
        adapter supports them, and finishes via a reserved `submit(...)` tool whose arguments mirror
        the module's output signature.
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.parallel_tool_calls = parallel_tool_calls

        tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
        self.tools = {tool.name: tool for tool in tools}
        self._validate_tool_names()
        self.submit_tool = self._build_submit_tool()
        self.output_signature = self._build_output_signature(signature)

        action_signature = self._build_action_signature(signature)
        extract_signature = self._build_extract_signature(signature)

        self.react = dspy.Predict(action_signature)
        self.extract = dspy.ChainOfThought(extract_signature)

    def _validate_tool_names(self) -> None:
        if self._SUBMIT_TOOL_NAME in self.tools:
            raise ValueError(f"Tool name '{self._SUBMIT_TOOL_NAME}' is reserved by dspy.ReAct.")

    def _build_action_signature(self, signature: type["Signature"]) -> type["Signature"]:
        inputs = ", ".join(f"`{name}`" for name in signature.input_fields.keys())
        outputs = ", ".join(f"`{name}`" for name in signature.output_fields.keys())
        instructions = [f"{signature.instructions}\n"] if signature.instructions else []
        instructions.extend(
            [
                "You are an agent acting over a running chat transcript in `history`.",
                "The latest user message in `history` is the current request.",
                f"Your job is to use the available tools to gather whatever is needed to produce {outputs}.",
                "You may optionally include a short assistant message in `next_message` before or alongside tool calls.",
                f"When you are fully ready to answer the user, call `{self._SUBMIT_TOOL_NAME}` with exactly the output fields.",
                f"Do not provide final {outputs} directly in `next_message`; use `{self._SUBMIT_TOOL_NAME}`.",
                f"Do not combine `{self._SUBMIT_TOOL_NAME}` with any other tool call in the same turn.",
            ]
        )
        if inputs:
            instructions.append(
                f"The original request fields were {inputs}, but the current request is already materialized inside `history`."
            )

        return (
            dspy.Signature({}, "\n".join(instructions))
            .append("history", dspy.InputField(), type_=dspy.History)
            .append("tools", dspy.InputField(), type_=list[dspy.Tool])
            .append("next_message", dspy.OutputField(), type_=str)
            .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
        )

    def _build_extract_signature(self, signature: type["Signature"]) -> type["Signature"]:
        extract_signature = dspy.Signature({}, signature.instructions).append("history", dspy.InputField(), type_=History)
        for name, field in signature.output_fields.items():
            extract_signature = extract_signature.append(name, dspy.OutputField(desc=field.json_schema_extra["desc"]), type_=field.annotation)
        return extract_signature

    def _build_output_signature(self, signature: type["Signature"]) -> type["Signature"]:
        output_signature = dspy.Signature({}, signature.instructions)
        for name, field in signature.output_fields.items():
            output_signature = output_signature.append(name, dspy.OutputField(desc=field.json_schema_extra["desc"]), type_=field.annotation)
        return output_signature

    def _build_submit_tool(self) -> Tool:
        output_names = ", ".join(f"`{name}`" for name in self.signature.output_fields.keys())
        args = {}
        arg_types = {}
        for name, field in self.signature.output_fields.items():
            arg_types[name] = field.annotation
            args[name] = _resolve_json_schema_reference(TypeAdapter(field.annotation).json_schema())
        return Tool(
            func=lambda **kwargs: kwargs,
            name=self._SUBMIT_TOOL_NAME,
            desc=f"Submit the final outputs for the task. Provide exactly these fields: {output_names}.",
            args=args,
            arg_types=arg_types,
        )

    def forward(self, **input_args):
        max_iters = input_args.pop("max_iters", self.max_iters)
        history = self._coerce_history(input_args)

        lm, adapter = self._get_lm_and_adapter(self.react)
        user_message = self._materialize_user_message(input_args, adapter)
        history = history.with_messages([user_message], lm=lm, adapter=adapter)

        for _ in range(max_iters):
            try:
                pred = self._call_with_potential_history_compaction(
                    self.react,
                    transcript=history,
                    history=history,
                    tools=self._get_runtime_tools(),
                )
            except ValueError as err:
                logger.warning("Ending the trajectory after history compaction retries failed: %s", err)
                break

            tool_calls = getattr(pred, "tool_calls", None)
            next_message = getattr(pred, "next_message", None)
            if tool_calls is None or not tool_calls.tool_calls:
                if next_message:
                    history = history.with_messages([{"role": "assistant", "content": next_message}], lm=lm, adapter=adapter)
                break

            assistant_message, tool_call_ids = self._build_assistant_tool_call_message(history, pred)
            submit_indices = [
                idx for idx, tool_call in enumerate(tool_calls.tool_calls) if tool_call.name == self._SUBMIT_TOOL_NAME
            ]
            if submit_indices:
                history = self._handle_submit_turn(history, assistant_message, tool_calls, tool_call_ids, lm=lm, adapter=adapter)
                if self._has_successful_submit(history, tool_call_ids[submit_indices[0]]):
                    outputs = self._extract_submit_outputs_from_history(history, tool_call_ids[submit_indices[0]])
                    return dspy.Prediction(history=history, trajectory=self._history_to_trajectory(history), **outputs)
                continue

            tool_messages = self._execute_tool_batch_sync(tool_calls, tool_call_ids)
            history = history.with_messages([assistant_message, *tool_messages], lm=lm, adapter=adapter)

        extract_prediction = self._call_with_potential_history_compaction(
            self.extract,
            transcript=history,
            history=history,
        )
        extracted_outputs = {name: getattr(extract_prediction, name) for name in self.signature.output_fields.keys()}
        history = history.with_messages(
            self._build_synthetic_submit_completion(history, extracted_outputs),
            lm=lm,
            adapter=adapter,
        )
        return dspy.Prediction(history=history, trajectory=self._history_to_trajectory(history), **extract_prediction)

    async def aforward(self, **input_args):
        max_iters = input_args.pop("max_iters", self.max_iters)
        history = self._coerce_history(input_args)

        lm, adapter = self._get_lm_and_adapter(self.react)
        user_message = self._materialize_user_message(input_args, adapter)
        history = history.with_messages([user_message], lm=lm, adapter=adapter)

        for _ in range(max_iters):
            try:
                pred = await self._async_call_with_potential_history_compaction(
                    self.react,
                    transcript=history,
                    history=history,
                    tools=self._get_runtime_tools(),
                )
            except ValueError as err:
                logger.warning("Ending the trajectory after history compaction retries failed: %s", err)
                break

            tool_calls = getattr(pred, "tool_calls", None)
            next_message = getattr(pred, "next_message", None)
            if tool_calls is None or not tool_calls.tool_calls:
                if next_message:
                    history = history.with_messages([{"role": "assistant", "content": next_message}], lm=lm, adapter=adapter)
                break

            assistant_message, tool_call_ids = self._build_assistant_tool_call_message(history, pred)
            submit_indices = [
                idx for idx, tool_call in enumerate(tool_calls.tool_calls) if tool_call.name == self._SUBMIT_TOOL_NAME
            ]
            if submit_indices:
                history = self._handle_submit_turn(history, assistant_message, tool_calls, tool_call_ids, lm=lm, adapter=adapter)
                if self._has_successful_submit(history, tool_call_ids[submit_indices[0]]):
                    outputs = self._extract_submit_outputs_from_history(history, tool_call_ids[submit_indices[0]])
                    return dspy.Prediction(history=history, trajectory=self._history_to_trajectory(history), **outputs)
                continue

            tool_messages = await self._execute_tool_batch_async(tool_calls, tool_call_ids)
            history = history.with_messages([assistant_message, *tool_messages], lm=lm, adapter=adapter)

        extract_prediction = await self._async_call_with_potential_history_compaction(
            self.extract,
            transcript=history,
            history=history,
        )
        extracted_outputs = {name: getattr(extract_prediction, name) for name in self.signature.output_fields.keys()}
        history = history.with_messages(
            self._build_synthetic_submit_completion(history, extracted_outputs),
            lm=lm,
            adapter=adapter,
        )
        return dspy.Prediction(history=history, trajectory=self._history_to_trajectory(history), **extract_prediction)

    def _coerce_history(self, input_args: dict[str, Any]) -> History:
        history = input_args.pop("history", None)
        trajectory = input_args.pop("trajectory", None)

        if history is not None:
            source = history
        else:
            source = trajectory

        if source is None:
            return History.raw()
        if isinstance(source, History):
            if source.mode == "raw":
                return source
            return History.raw(messages=source.messages)
        if isinstance(source, dict):
            return self._legacy_trajectory_to_history(source)

        raise TypeError("`history` must be a dspy.History, and `trajectory` must be a dict or dspy.History.")

    def _get_runtime_tools(self) -> list[Tool]:
        return [*self.tools.values(), self.submit_tool]

    def _get_lm_and_adapter(self, module: Any | None = None) -> tuple["LM | None", "Adapter"]:
        lm = getattr(module, "lm", None) or dspy.settings.lm
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        return lm, adapter

    def _call_with_potential_history_compaction(self, module, transcript: History, **module_inputs):
        lm, adapter = self._get_lm_and_adapter(module)
        current_history = transcript
        for _ in range(3):
            try:
                return module(**module_inputs)
            except ContextWindowExceededError:
                logger.warning("History exceeded the context window, compacting more aggressively.")
                current_history = self._aggressively_compact_history(current_history, lm=lm, adapter=adapter)
                module_inputs["history"] = current_history
        raise ValueError("The context window was exceeded even after 3 attempts to compact the history.")

    async def _async_call_with_potential_history_compaction(self, module, transcript: History, **module_inputs):
        lm, adapter = self._get_lm_and_adapter(module)
        current_history = transcript
        for _ in range(3):
            try:
                return await module.acall(**module_inputs)
            except ContextWindowExceededError:
                logger.warning("History exceeded the context window, compacting more aggressively.")
                current_history = self._aggressively_compact_history(current_history, lm=lm, adapter=adapter)
                module_inputs["history"] = current_history
        raise ValueError("The context window was exceeded even after 3 attempts to compact the history.")

    def _aggressively_compact_history(self, history: History, *, lm: "LM | None", adapter: "Adapter") -> History:
        if history.mode != "raw" or history.compaction is None:
            raise ValueError("The history exceeded the context window and cannot be compacted further.")

        keep_last_messages = max(1, history.compaction.keep_last_messages - 1)
        max_visible_tokens = max(1, history.compaction.max_visible_tokens // 2)
        compacted = History(
            messages=history.messages,
            mode="raw",
            compaction=HistoryCompaction(
                max_visible_tokens=max_visible_tokens,
                keep_last_messages=keep_last_messages,
            ),
            summary=history.summary,
            compacted_count=history.compacted_count,
        )
        compacted = compacted.with_messages([], lm=lm, adapter=adapter)
        if compacted.compacted_count == history.compacted_count and compacted.summary == history.summary:
            raise ValueError("The history exceeded the context window and could not be compacted further.")
        return compacted

    def _materialize_user_message(self, input_args: dict[str, Any], adapter: "Adapter") -> dict[str, Any]:
        return {
            "role": "user",
            "content": adapter.format_user_message_content(self.signature, input_args),
        }

    def _materialize_final_assistant_message(self, outputs: dict[str, Any], adapter: "Adapter") -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": adapter.format_assistant_message_content(self.output_signature, outputs),
        }

    def _build_assistant_tool_call_message(self, history: History, pred: dspy.Prediction) -> tuple[dict[str, Any], list[str]]:
        tool_call_ids = []
        prior_tool_turns = sum(
            1
            for message in history.messages
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        tool_calls = []
        for idx, tool_call in enumerate(pred.tool_calls.tool_calls, start=1):
            tool_call_id = f"call_{prior_tool_turns + 1}_{idx}"
            tool_call_ids.append(tool_call_id)
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.args, ensure_ascii=False),
                    },
                }
            )

        assistant_message = {
            "role": "assistant",
            "content": pred.next_message or None,
            "tool_calls": tool_calls,
        }
        return assistant_message, tool_call_ids

    def _handle_submit_turn(
        self,
        history: History,
        assistant_message: dict[str, Any],
        tool_calls: ToolCalls,
        tool_call_ids: list[str],
        *,
        lm: "LM | None",
        adapter: "Adapter",
    ) -> History:
        if len(tool_calls.tool_calls) != 1:
            tool_messages = [
                self._make_tool_message(
                    tool_call_id=tool_call_id,
                    tool_name=tool_call.name,
                    content=f"Tool error in {tool_call.name}: `submit` cannot be combined with other tool calls.",
                )
                for tool_call, tool_call_id in zip(tool_calls.tool_calls, tool_call_ids, strict=False)
            ]
            return history.with_messages([assistant_message, *tool_messages], lm=lm, adapter=adapter)

        submit_call = tool_calls.tool_calls[0]
        submit_outputs, error = self._parse_submit_outputs(submit_call.args)
        if error:
            tool_message = self._make_tool_message(
                tool_call_id=tool_call_ids[0],
                tool_name=self._SUBMIT_TOOL_NAME,
                content=error,
            )
            return history.with_messages([assistant_message, tool_message], lm=lm, adapter=adapter)

        final_assistant_message = self._materialize_final_assistant_message(submit_outputs, adapter)
        submit_tool_message = self._make_tool_message(
            tool_call_id=tool_call_ids[0],
            tool_name=self._SUBMIT_TOOL_NAME,
            content=f"Submitted final outputs successfully: {json.dumps(serialize_for_json(submit_outputs), ensure_ascii=False)}",
        )
        return history.with_messages(
            [assistant_message, submit_tool_message, final_assistant_message],
            lm=lm,
            adapter=adapter,
        )

    def _parse_submit_outputs(self, raw_output: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(raw_output, dict):
            return None, f"Tool error in {self._SUBMIT_TOOL_NAME}: expected an object with output fields."

        output_field_names = list(self.signature.output_fields.keys())
        missing = set(output_field_names) - set(raw_output)
        extra = set(raw_output) - set(output_field_names)
        if missing:
            return None, (
                f"Tool error in {self._SUBMIT_TOOL_NAME}: missing output fields {sorted(missing)}. "
                f"Use {self._SUBMIT_TOOL_NAME}({', '.join(output_field_names)})."
            )
        if extra:
            return None, f"Tool error in {self._SUBMIT_TOOL_NAME}: unexpected output fields {sorted(extra)}."

        parsed_outputs = {}
        type_errors = []
        for name, field in self.signature.output_fields.items():
            try:
                parsed_outputs[name] = parse_value(raw_output[name], field.annotation)
            except (ValueError, pydantic.ValidationError) as err:
                type_errors.append(f"{name}: {err}")

        if type_errors:
            return None, f"Tool error in {self._SUBMIT_TOOL_NAME}: " + "; ".join(type_errors)

        return parsed_outputs, None

    def _has_successful_submit(self, history: History, tool_call_id: str) -> bool:
        for message in reversed(history.messages):
            if message.get("role") != "tool":
                continue
            if message.get("tool_call_id") != tool_call_id:
                continue
            return "Submitted final outputs successfully:" in str(message.get("content"))
        return False

    def _extract_submit_outputs_from_history(self, history: History, tool_call_id: str) -> dict[str, Any]:
        for message in history.messages:
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            for tool_call in message["tool_calls"]:
                if tool_call["id"] != tool_call_id:
                    continue
                submit_outputs, error = self._parse_submit_outputs(json.loads(tool_call["function"]["arguments"]))
                if error:
                    raise ValueError(error)
                return submit_outputs
        raise ValueError("Could not recover submitted outputs from the history.")

    def _build_synthetic_submit_completion(self, history: History, outputs: dict[str, Any]) -> list[dict[str, Any]]:
        _, adapter = self._get_lm_and_adapter(self.extract)
        prior_tool_turns = sum(
            1
            for message in history.messages
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        tool_call_id = f"call_{prior_tool_turns + 1}_1"
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": self._SUBMIT_TOOL_NAME,
                        "arguments": json.dumps(serialize_for_json(outputs), ensure_ascii=False),
                    },
                }
            ],
        }
        tool_message = self._make_tool_message(
            tool_call_id=tool_call_id,
            tool_name=self._SUBMIT_TOOL_NAME,
            content=f"Submitted final outputs successfully: {json.dumps(serialize_for_json(outputs), ensure_ascii=False)}",
        )
        final_assistant_message = self._materialize_final_assistant_message(outputs, adapter)
        return [assistant_message, tool_message, final_assistant_message]

    def _execute_tool_batch_sync(self, tool_calls: ToolCalls, tool_call_ids: list[str]) -> list[dict[str, Any]]:
        if self.parallel_tool_calls and len(tool_calls.tool_calls) > 1:
            executor = ParallelExecutor(
                num_threads=len(tool_calls.tool_calls),
                disable_progress_bar=True,
                straggler_limit=0,
            )
            return executor.execute(
                lambda item: self._execute_single_tool_call_sync(item[0], item[1]),
                list(zip(tool_calls.tool_calls, tool_call_ids, strict=False)),
            )
        return [
            self._execute_single_tool_call_sync(tool_call, tool_call_id)
            for tool_call, tool_call_id in zip(tool_calls.tool_calls, tool_call_ids, strict=False)
        ]

    async def _execute_tool_batch_async(self, tool_calls: ToolCalls, tool_call_ids: list[str]) -> list[dict[str, Any]]:
        coroutines = [
            self._execute_single_tool_call_async(tool_call, tool_call_id)
            for tool_call, tool_call_id in zip(tool_calls.tool_calls, tool_call_ids, strict=False)
        ]
        if self.parallel_tool_calls and len(coroutines) > 1:
            return list(await asyncio.gather(*coroutines))
        return [await coroutine for coroutine in coroutines]

    def _execute_single_tool_call_sync(self, tool_call: ToolCalls.ToolCall, tool_call_id: str) -> dict[str, Any]:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return self._make_tool_message(
                tool_call_id=tool_call_id,
                tool_name=tool_call.name,
                content=f"Tool error in {tool_call.name}: unknown tool `{tool_call.name}`.",
            )

        try:
            observation = tool(**tool_call.args)
        except ValueError as err:
            return self._make_tool_message(
                tool_call_id=tool_call_id,
                tool_name=tool_call.name,
                content=f"Tool error in {tool_call.name}: {err}",
            )

        return self._make_tool_message(
            tool_call_id=tool_call_id,
            tool_name=tool_call.name,
            content=self._format_observation_content(observation),
        )

    async def _execute_single_tool_call_async(self, tool_call: ToolCalls.ToolCall, tool_call_id: str) -> dict[str, Any]:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return self._make_tool_message(
                tool_call_id=tool_call_id,
                tool_name=tool_call.name,
                content=f"Tool error in {tool_call.name}: unknown tool `{tool_call.name}`.",
            )

        try:
            observation = await tool.acall(**tool_call.args)
        except ValueError as err:
            return self._make_tool_message(
                tool_call_id=tool_call_id,
                tool_name=tool_call.name,
                content=f"Tool error in {tool_call.name}: {err}",
            )

        return self._make_tool_message(
            tool_call_id=tool_call_id,
            tool_name=tool_call.name,
            content=self._format_observation_content(observation),
        )

    def _make_tool_message(self, *, tool_call_id: str, tool_name: str, content: Any) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        }

    def _format_observation_content(self, observation: Any) -> Any:
        if isinstance(observation, str):
            return observation
        if isinstance(observation, (list, tuple)):
            return "\n".join(self._format_observation_part(item) for item in observation)

        jsonable = serialize_for_json(observation)
        if isinstance(jsonable, (dict, list)):
            return json.dumps(jsonable, ensure_ascii=False)
        return str(jsonable)

    def _format_observation_part(self, observation: Any) -> str:
        jsonable = serialize_for_json(observation)
        if isinstance(jsonable, (dict, list)):
            return json.dumps(jsonable, ensure_ascii=False)
        return str(jsonable)

    def _history_to_trajectory(self, history: History) -> dict[str, Any]:
        trajectory = {}
        step_idx = 0
        messages = history.messages
        for idx, message in enumerate(messages):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue

            tool_calls = message["tool_calls"]
            tool_messages = []
            tool_message_idx = idx + 1
            while tool_message_idx < len(messages) and messages[tool_message_idx].get("role") == "tool":
                tool_messages.append(messages[tool_message_idx])
                tool_message_idx += 1

            tool_message_by_id = {
                tool_message.get("tool_call_id"): tool_message.get("content")
                for tool_message in tool_messages
            }
            names = [tool_call["function"]["name"] for tool_call in tool_calls]
            args = [json.loads(tool_call["function"]["arguments"]) for tool_call in tool_calls]
            observations = [tool_message_by_id.get(tool_call["id"]) for tool_call in tool_calls]

            trajectory[f"thought_{step_idx}"] = message.get("content")
            trajectory[f"tool_name_{step_idx}"] = names[0] if len(names) == 1 else names
            trajectory[f"tool_args_{step_idx}"] = args[0] if len(args) == 1 else args
            trajectory[f"observation_{step_idx}"] = observations[0] if len(observations) == 1 else observations
            step_idx += 1

        return trajectory

    def _legacy_trajectory_to_history(self, trajectory: dict[str, Any]) -> History:
        messages = []
        idx = 0
        while f"tool_name_{idx}" in trajectory:
            thought = trajectory.get(f"thought_{idx}")
            tool_name = trajectory[f"tool_name_{idx}"]
            tool_args = trajectory.get(f"tool_args_{idx}", {})
            observation = trajectory.get(f"observation_{idx}")

            if tool_name == "finish":
                if thought:
                    messages.append({"role": "assistant", "content": thought})
                idx += 1
                continue

            tool_call_id = f"legacy_call_{idx}"
            messages.append(
                {
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            messages.append(
                self._make_tool_message(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    content=self._format_observation_content(observation),
                )
            )
            idx += 1

        return History.raw(messages=messages)

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        current_trajectory = dict(trajectory)
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(current_trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                current_trajectory = self.truncate_trajectory(current_trajectory)
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        current_trajectory = dict(trajectory)
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(current_trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                current_trajectory = self.truncate_trajectory(current_trajectory)
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

    def truncate_trajectory(self, trajectory):
        keys = list(trajectory.keys())
        if len(keys) < 4:
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory
