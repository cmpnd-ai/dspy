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

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend([
            f"You are an Agent. Given {inputs}, use tools to produce {outputs}.",
            "Each turn: think, then call one or more tools. After each tool call you receive an observation.",
            "When you have enough information to answer, call `submit` to finish. Do not keep using tools after you have the answer.\n",
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
            "Each turn: think, then call one or more tools. After each tool call you receive an observation.",
            "When you have enough information to answer, call `submit` to finish. Do not keep using tools after you have the answer.\n",
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
        # Bypass self.react so the directive is the LAST message the model sees.
        # self.react would append a user message after our directive, drowning it out.
        import json_repair

        lm = dspy.settings.lm
        adapter = dspy.settings.adapter or dspy.ChatAdapter()

        # Replicate the same preprocessing that Predict.forward / adapter.__call__ would do.
        signature = self.react.signature
        demos = self.react.demos
        tool_list = list(self.tools.values())
        inputs = {**input_args, "history": history, "tools": tool_list}

        lm_kwargs = {**self.react.config}
        processed_sig = adapter._call_preprocess(lm, lm_kwargs, signature, inputs)
        messages = adapter.format(processed_sig, demos, inputs)

        # Build the directive that tells the model to call submit NOW.
        outputs = ", ".join([f"`{k}`" for k in self.signature.output_fields.keys()])
        directive = (
            f"You have used all your allowed iterations. You MUST call the `submit` tool now "
            f"with {outputs} based on the information you have gathered so far. "
            f"Do not call any other tool. Call submit immediately."
        )

        # Replace the last user message with our directive so it's the final
        # thing the model sees before generating.
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i] = {"role": "user", "content": directive}
                break

        # When native function calling is active, mechanically force the submit tool
        # so reasoning models can't ignore the directive via their internal CoT.
        if "tools" in lm_kwargs:
            lm_kwargs["tool_choice"] = {"type": "function", "function": {"name": "submit"}}

        # Call the LM directly.
        try:
            raw_outputs = lm(messages=messages, **lm_kwargs)
        except Exception:
            # Provider may not support tool_choice; retry without it
            lm_kwargs.pop("tool_choice", None)
            try:
                raw_outputs = lm(messages=messages, **lm_kwargs)
            except Exception:
                return dspy.Prediction(history=history)

        if not raw_outputs or not isinstance(raw_outputs, list):
            return dspy.Prediction(history=history)

        # Parse tool_calls from the first completion.
        output = raw_outputs[0]
        tool_calls = None
        if isinstance(output, dict):
            tool_calls = output.get("tool_calls")

        # --- Native FC path: tool_calls present in the response dict ---
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("function", {}).get("name")
                if name == "submit":
                    args_raw = tc.get("function", {}).get("arguments", "{}")
                    args = json_repair.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    try:
                        result = self.tools["submit"](**args)
                        history.append_final(result)
                        return dspy.Prediction(history=history, **result)
                    except Exception:
                        pass
            return dspy.Prediction(history=history)

        # --- Non-native text path: parse the text response for a submit call ---
        text = output if isinstance(output, str) else (output.get("text", "") if isinstance(output, dict) else "")
        if text:
            # Try to parse tool_calls from the text using the adapter
            try:
                parsed = adapter.parse(processed_sig, text)
                tc_obj = parsed.get("tool_calls")
                if tc_obj and hasattr(tc_obj, "tool_calls"):
                    for tool_call in tc_obj.tool_calls:
                        if tool_call.name == "submit":
                            try:
                                result = self.tools["submit"](**(tool_call.args or {}))
                                history.append_final(result)
                                return dspy.Prediction(history=history, **result)
                            except Exception:
                                pass
            except Exception:
                pass

            # Last resort: try to extract output field values from the text
            # using the original task signature (e.g. "question -> answer").
            try:
                parsed = adapter.parse(self.signature, text)
                if any(v is not None for v in parsed.values()):
                    history.append_final(parsed)
                    return dspy.Prediction(history=history, **parsed)
            except Exception:
                pass

        return dspy.Prediction(history=history)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    import traceback
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
