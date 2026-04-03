"""
Major updates in ReActV2:
Native and parallel tool calling + tool history.
Compaction.
Finish -> submit. No more extract
Optimizing tool descriptions?

ReActV2 Things to test:
- multiple Parallel Tool calls
- what happens if native tool calling is disabled but Tools are passed in
- poorly formatted tool calls
json vs chat adapter

history 
- handle images?
- serialize + deserialize history?
"""

import logging
from typing import TYPE_CHECKING, Callable

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class ReActV2(Module):
    def __init__(self, signature: type["Signature"] | str, tools: list[Callable], max_iters: int = 20):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

        Examples:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct(signature="question->answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        # TODO: Modify for parallel and native tool calls
        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["submit"] = Tool(
            func=lambda: "Completed.", # TODO: And this is now validation on the outputs, we dont necessarily just exit anymore
            name="submit",
            desc=f"Submit the outputs for the the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={}, # TODO: make this take the output args, should raise an error if the outputs are not provided properly
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("history", dspy.InputField(), type_=dspy.History)
            .append("tools", dspy.InputField(), type_=list[dspy.Tool])
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
        )

        self.tools = tools
        self.react = dspy.Predict(react_signature)


    def forward(self, **input_args):
        history = input_args.pop("history", dspy.History(messages=[]))
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                history.compact_if_needed()
                pred: dspy.Prediction = self.react(history=history, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the history: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            observations: list[tuple[str, bool]] = []
            for tool_call in pred.tool_calls.tool_calls:
                # TODO: make this actually parallel
                try:
                    observation = self.tools[tool_call.name](**tool_call.args)
                    observations.append((observation, False))
                except Exception as err:
                    observation = f"Execution error in {tool_call.name}: {_fmt_exc(err)}"
                    observations.append((observation, True))

            print(dspy.inspect_history())

            # PAY ATTENTION: This is the place to focus on
            # this becomes either a native tool call and a native tool result, or a fake tool call and a user message result
            # OH this is a weird one because we dont want the user message to be the last thing that a model sees.
            history.add_message(signature=self.react.signature, inputs=input_args, prediction=pred, tool_observations=observations) # this should have the adapter formatting + add history to input_args

            for tool_call, (result, did_err) in zip(pred.tool_calls.tool_calls, observations):
                # we could also isinstance check for a prediction
                if tool_call.name == "submit" and not did_err:
                    return dspy.Prediction(history=history, **result) # result is of type dict[str, Any] but we are guarateed that it matches our output fields

    # async def aforward(self, **input_args):
    #     trajectory = {}
    #     max_iters = input_args.pop("max_iters", self.max_iters)
    #     for idx in range(max_iters):
    #         try:
    #             pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
    #         except ValueError as err:
    #             logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
    #             break

    #         trajectory[f"thought_{idx}"] = pred.next_thought
    #         trajectory[f"tool_name_{idx}"] = pred.next_tool_name
    #         trajectory[f"tool_args_{idx}"] = pred.next_tool_args

    #         try:
    #             trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
    #         except Exception as err:
    #             trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

    #         if pred.next_tool_name == "finish":
    #             break

    #     extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
    #     return dspy.Prediction(trajectory=trajectory, **extract)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


"""
Thoughts and Planned Improvements for dspy.ReAct.

TOPIC 01: How Trajectories are Formatted, or rather when they are formatted.

Right now, both sub-modules are invoked with a `trajectory` argument, which is a string formatted in `forward`. Though
the formatter uses a general adapter.format_fields, the tracing of DSPy only sees the string, not the formatting logic.

What this means is that, in demonstrations, even if the user adjusts the adapter for a fixed program, the demos' format
will not update accordingly, but the inference-time trajectories will.

One way to fix this is to support `format=fn` in the dspy.InputField() for "trajectory" in the signatures. But this
means that care must be taken that the adapter is accessed at `forward` runtime, not signature definition time.

Another potential fix is to more natively support a "variadic" input field, where the input is a list of dictionaries,
or a big dictionary, and have each adapter format it accordingly.

Trajectories also affect meta-programming modules that view the trace later. It's inefficient O(n^2) to view the
trace of every module repeating the prefix.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?


TOPIC 06: Idiomatically allowing tools that maintain state across iterations, but not across different `forward` calls.
    * So the tool would be newly initialized at the start of each `forward` call, but maintain state across iterations.
    * This is pretty useful for allowing the agent to keep notes or count certain things, etc.
"""
