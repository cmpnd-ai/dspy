import logging
from typing import TYPE_CHECKING, Any, Callable

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


def _build_submit_tool(signature: type["Signature"]) -> Tool:
    outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
    output_args = {}
    output_arg_types = {}
    for k, v in signature.output_fields.items():
        output_args[k] = {"type": "string"}
        output_arg_types[k] = v.annotation if hasattr(v, "annotation") else str

    return Tool(
        func=lambda **kwargs: kwargs,
        name="submit",
        desc=f"Submit the final outputs ({outputs}) for the task.",
        args=output_args,
        arg_types=output_arg_types,
    )


class ReActV2(Module):
    def __init__(self, signature: type["Signature"] | str, tools: list[Callable], max_iters: int = 20):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}
        tools["submit"] = _build_submit_tool(signature)

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend([
            f"You are an Agent. Given {inputs}, use tools to produce {outputs}.",
            "Each turn: think, then call a tool. After each tool call you receive an observation.",
            "When you have enough information, call `submit` with the output fields.\n",
            "Available tools:\n",
        ])

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("history", dspy.InputField(), type_=dspy.History)
            .append("tools", dspy.InputField(), type_=list[dspy.Tool])
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
        )

        self.tools = tools
        self.react = dspy.Predict(react_signature)

    def _rebuild_instructions(self):
        """Regenerate the instruction string from current tool descs.

        Called after GEPA updates tool.desc so that both text-mode prompts
        and native FC schemas reflect the optimized descriptions.
        """
        inputs = ", ".join([f"`{k}`" for k in self.signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in self.signature.output_fields.keys()])
        instr = [f"{self.signature.instructions}\n"] if self.signature.instructions else []

        instr.extend([
            f"You are an Agent. Given {inputs}, use tools to produce {outputs}.",
            "Each turn: think, then call a tool. After each tool call you receive an observation.",
            "When you have enough information, call `submit` with the output fields.\n",
            "Available tools:\n",
        ])

        for idx, tool in enumerate(self.tools.values()):
            instr.append(f"({idx + 1}) {tool}")

        self.react.signature = self.react.signature.with_instructions("\n".join(instr))

    def forward(self, **input_args):
        history = input_args.pop("history", dspy.History(messages=[]))
        max_iters = input_args.pop("max_iters", self.max_iters)
        tool_list = list(self.tools.values())

        if not history.has_open_episode():
            history.append_input(input_args)

        for idx in range(max_iters):
            history.compact_if_needed()
            try:
                pred: dspy.Prediction = self.react(history=history, tools=tool_list, **input_args)
            except (AdapterParseError, ValueError) as err:
                logger.warning(f"Agent iteration {idx} failed: {_fmt_exc(err)}")
                break

            if pred.tool_calls is None or not pred.tool_calls.tool_calls:
                logger.warning("Agent returned no tool calls, ending loop.")
                break

            observations: list[tuple[Any, bool]] = []
            for tool_call in pred.tool_calls.tool_calls:
                tool = self.tools.get(tool_call.name)
                if tool is None:
                    observations.append((f"Unknown tool: {tool_call.name}", True))
                    continue
                try:
                    result = tool(**tool_call.args)
                    observations.append((result, False))
                except Exception as err:
                    observations.append((f"Execution error in {tool_call.name}: {_fmt_exc(err)}", True))

            history.append_action(
                thought=pred.next_thought,
                tool_calls=pred.tool_calls,
                observations=observations,
            )

            for tool_call, (result, did_err) in zip(pred.tool_calls.tool_calls, observations):
                if tool_call.name == "submit" and not did_err:
                    history.append_final(result)
                    return dspy.Prediction(history=history, **result)

        # Forced submit: ask the model to submit one more time
        return self._forced_submit(history, input_args)

    def _forced_submit(self, history, input_args):
        try:
            pred = self.react(history=history, tools=list(self.tools.values()), **input_args)
        except (AdapterParseError, ValueError):
            return dspy.Prediction(history=history)

        if pred.tool_calls is None or not pred.tool_calls.tool_calls:
            return dspy.Prediction(history=history)

        for tool_call in pred.tool_calls.tool_calls:
            if tool_call.name == "submit":
                tool = self.tools["submit"]
                try:
                    result = tool(**tool_call.args)
                    return dspy.Prediction(history=history, **result)
                except Exception:
                    pass
        return dspy.Prediction(history=history)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    import traceback
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
