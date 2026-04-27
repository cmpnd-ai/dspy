import logging
import traceback
from typing import TYPE_CHECKING, Callable

import dspy
from dspy.adapters.types.history import ActionEvent, InputEvent, Observation
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
        desc=f"Call this tool to end the task and return your final answer. Takes: {outputs}.",
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
        self.tools = tools

        react_signature = (
            dspy.Signature({**signature.input_fields}, self._build_instructions())
            .append("history", dspy.InputField(), type_=dspy.History)
            .append("tools", dspy.InputField(), type_=list[dspy.Tool])
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
        )

        # Extract fallback: dedicated LM call to extract answer from history
        # (like v1's self.extract, fires only when submit fails in _forced_submit)
        extract_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(desc="The agent's history of thoughts, actions, and observations"), type_=str)

        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(extract_signature)

    def _build_instructions(self):
        """Build the instruction string from current signature and tools."""
        inputs = ", ".join([f"`{k}`" for k in self.signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in self.signature.output_fields.keys()])
        instr = [f"{self.signature.instructions}\n"] if self.signature.instructions else []

        instr.extend([
            f"You are an Agent. Given {inputs}, use tools to produce {outputs}.",
            "Each turn: think, then call one or more tools. After each tool call you receive an observation.",
            "When you have enough information to answer, call `submit` to finish. Do not keep using tools after you have the answer.\n",
            "Available tools:\n",
        ])

        for idx, tool in enumerate(self.tools.values()):
            instr.append(f"({idx + 1}) {tool}")

        return "\n".join(instr)

    def _rebuild_instructions(self):
        """Regenerate the instruction string from current tool descs.

        Called after GEPA updates tool.desc so that both text-mode prompts
        and native FC schemas reflect the optimized descriptions.
        """
        base_instr = self._build_instructions()
        self.react.signature = self.react.signature.with_instructions(base_instr)

    def forward(self, **input_args):
        # Callers can pass a History with a compact_fn for automatic compaction each iteration.
        history = input_args.pop("history", dspy.History(messages=[]))
        max_iters = input_args.pop("max_iters", self.max_iters)
        tool_list = list(self.tools.values())

        if not history.has_open_episode():
            history.append_input(input_args)

        break_reason = None
        for idx in range(max_iters):
            try:
                pred: dspy.Prediction = self.react(history=history, tools=tool_list, **input_args)
            except (AdapterParseError, ValueError) as err:
                logger.warning(f"Agent iteration {idx} failed: {_fmt_exc(err)}")
                break_reason = "parse_error"
                break

            if pred.tool_calls is None or not pred.tool_calls.tool_calls:
                logger.warning("Agent returned no tool calls, ending loop.")
                break_reason = "no_tool_calls"
                break

            observations: list[Observation] = []
            for tool_call in pred.tool_calls.tool_calls:
                tool = self.tools.get(tool_call.name)
                if tool is None:
                    observations.append(Observation(value=f"Unknown tool: {tool_call.name}", is_error=True))
                    continue
                try:
                    result = tool(**tool_call.args)
                    observations.append(Observation(value=result, is_error=False))
                except Exception as err:
                    observations.append(Observation(value=f"Execution error in {tool_call.name}: {_fmt_exc(err)}", is_error=True))

            history.append_action(
                thought=pred.next_thought,
                tool_calls=pred.tool_calls,
                observations=observations,
            )

            for tool_call, obs in zip(pred.tool_calls.tool_calls, observations):
                if tool_call.name == "submit" and not obs.is_error:
                    history.append_output(obs.value)
                    return dspy.Prediction(history=history, termination_reason="submit", **obs.value)

        # Forced submit: ask the model to submit one more time
        return self._forced_submit(history, input_args, break_reason=break_reason)

    def _forced_submit(self, history, input_args, break_reason=None):
        tool_list = list(self.tools.values())

        # Tier 1: Re-use self.react with tool_choice forced to submit.
        adapter = dspy.settings.adapter
        native_fc = getattr(adapter, "use_native_function_calling", False) if adapter else False

        saved_config = dict(self.react.config)
        if native_fc:
            self.react.config["tool_choice"] = {"type": "function", "function": {"name": "submit"}}

        try:
            pred = self.react(history=history, tools=tool_list, **input_args)
        except Exception:
            pred = None
        finally:
            self.react.config.clear()
            self.react.config.update(saved_config)

        if pred and pred.tool_calls and pred.tool_calls.tool_calls:
            for tool_call in pred.tool_calls.tool_calls:
                if tool_call.name == "submit":
                    try:
                        result = self.tools["submit"](**tool_call.args)
                        history.append_action(
                            thought=pred.next_thought,
                            tool_calls=pred.tool_calls,
                            observations=[Observation(value=result, is_error=False)],
                        )
                        history.append_output(result)
                        return dspy.Prediction(history=history, termination_reason="forced_submit", **result)
                    except Exception:
                        pass

        # Tier 2: Extract fallback via ChainOfThought.
        try:
            trajectory_text = self._render_history_as_text(history)
            extract = self.extract(trajectory=trajectory_text, **input_args)
            result = {k: getattr(extract, k) for k in self.signature.output_fields if hasattr(extract, k)}
            if any(v is not None for v in result.values()):
                history.append_output(result)
                return dspy.Prediction(history=history, termination_reason="extract", **result)
        except Exception:
            pass

        return dspy.Prediction(history=history, termination_reason=break_reason or "failed")

    @staticmethod
    def _render_history_as_text(history) -> str:
        lines = []
        for event in history.messages:
            if isinstance(event, InputEvent):
                for k, v in event.inputs.items():
                    lines.append(f"[Input] {k}: {v}")
            elif isinstance(event, ActionEvent):
                if event.thought:
                    lines.append(f"[Thought] {event.thought}")
                if event.tool_calls and hasattr(event.tool_calls, "tool_calls"):
                    for i, tc in enumerate(event.tool_calls.tool_calls):
                        args_str = ", ".join(f"{k}={v!r}" for k, v in (tc.args or {}).items())
                        lines.append(f"[Action] {tc.name}({args_str})")
                        if i < len(event.observations):
                            obs_val, was_err = event.observations[i].value, event.observations[i].is_error
                            prefix = "[Error]" if was_err else "[Observation]"
                            lines.append(f"{prefix} {obs_val}")
        return "\n".join(lines)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
