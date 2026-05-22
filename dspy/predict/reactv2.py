import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import dspy
from dspy.adapters.types.history import History, Observation
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.exceptions import AdapterParseError, ContextWindowExceededError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

_ANNOTATION_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}


@dataclass(frozen=True)
class ToolObservation:
    value: object
    is_error: bool = False


def _build_submit_tool(signature: type["Signature"]) -> Tool:
    outputs = ", ".join([f"`{k}`" for k in signature.output_fields])
    output_args = {}
    output_arg_types = {}
    for name, field in signature.output_fields.items():
        annotation = getattr(field, "annotation", str)
        output_args[name] = {"type": _ANNOTATION_TO_JSON_TYPE.get(annotation, "string")}
        output_arg_types[name] = annotation

    return Tool(
        func=lambda **kwargs: kwargs,
        name="submit",
        desc=f"Call this tool to end the task and return your final answer. Takes: {outputs}.",
        args=output_args,
        arg_types=output_arg_types,
    )


class ReActV2(Module):
    def __init__(self, signature: type["Signature"] | str, tools: list[Callable], max_iters: int = 20):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
        self.tools = {tool.name: tool for tool in tools}
        self.tools["submit"] = _build_submit_tool(signature)

        react_signature = (
            dspy.Signature({**signature.input_fields}, self._build_instructions())
            .append("history", dspy.InputField(), type_=dspy.History)
            .append("tools", dspy.InputField(), type_=list[dspy.Tool])
            .append("next_thought", dspy.OutputField(), type_=dspy.Reasoning)
            .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
        )
        self.react = dspy.Predict(react_signature)

    def _build_instructions(self) -> str:
        inputs = ", ".join([f"`{k}`" for k in self.signature.input_fields])
        outputs = ", ".join([f"`{k}`" for k in self.signature.output_fields])
        instructions = [f"{self.signature.instructions}\n"] if self.signature.instructions else []
        instructions.extend(
            [
                f"You are an Agent. Given {inputs}, use tools to produce {outputs}.",
                "Each turn: think, then call one or more tools. After each tool call you receive an observation.",
                "When you have enough information to answer, call `submit` to finish.",
                "\nAvailable tools:\n",
            ]
        )
        instructions.extend(f"({idx + 1}) {tool}" for idx, tool in enumerate(self.tools.values()))
        return "\n".join(instructions)

    def forward(self, **input_args):
        history = input_args.pop("history", dspy.History(frames=[]))
        max_iters = input_args.pop("max_iters", self.max_iters)
        tool_list = list(self.tools.values())

        if not history.has_open_episode():
            history.append_input(input_args)

        break_reason = None
        for idx in range(max_iters):
            try:
                pred = self.react(history=history, tools=tool_list, **input_args)
            except ContextWindowExceededError:
                history.compact_if_needed()
                try:
                    pred = self.react(history=history, tools=tool_list, **input_args)
                except ContextWindowExceededError:
                    logger.warning("Context window exceeded after compaction, ending loop.")
                    break_reason = "context_overflow"
                    break
            except (AdapterParseError, ValueError) as err:
                logger.warning(f"Agent iteration {idx} failed: {_fmt_exc(err)}")
                break_reason = "parse_error"
                break

            if pred.tool_calls is None or not pred.tool_calls.tool_calls:
                logger.warning("Agent returned no tool calls, ending loop.")
                break_reason = "no_tool_calls"
                break

            tool_calls = pred.tool_calls.with_call_ids(f"call_{len(history.frames)}")
            observations = [self._execute_tool_call(tool_call) for tool_call in tool_calls.tool_calls]
            self._append_tool_turn(
                history,
                next_thought=pred.next_thought,
                tool_calls=tool_calls,
                observations=observations,
            )

            for tool_call, obs in zip(tool_calls.tool_calls, observations, strict=True):
                if tool_call.name == "submit" and not obs.is_error:
                    history.append_output(obs.value)
                    return dspy.Prediction(history=history, termination_reason="submit", **obs.value)

        return dspy.Prediction(history=history, termination_reason=break_reason or "max_iters")

    def _execute_tool_call(self, tool_call: ToolCalls.ToolCall) -> ToolObservation:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolObservation(value=f"Unknown tool: {tool_call.name}", is_error=True)
        try:
            return ToolObservation(value=tool(**tool_call.args), is_error=False)
        except Exception as err:
            return ToolObservation(value=f"Execution error in {tool_call.name}: {_fmt_exc(err)}", is_error=True)

    @staticmethod
    def _append_tool_turn(
        history: History,
        *,
        next_thought,
        tool_calls: ToolCalls,
        observations: list[ToolObservation],
    ) -> None:
        history.append_outputs(
            {"next_thought": next_thought, "tool_calls": tool_calls},
            observations=[
                Observation(
                    value=observation.value,
                    source="tool",
                    call_id=tool_call.id,
                    name=tool_call.name,
                    is_error=observation.is_error,
                )
                for tool_call, observation in zip(tool_calls.tool_calls, observations, strict=True)
            ],
        )


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
