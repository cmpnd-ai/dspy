"""Capability bridge and immutable policy for Monty invocations."""

from __future__ import annotations

import ast
import inspect
import keyword
import threading
from dataclasses import dataclass
from typing import Any

from pydantic_core import to_jsonable_python

from dspy.adapters.types.tool import Tool
from dspy.primitives.example import Example

RESERVED = {
    "dspy",
    "SUBMIT",
    "print",
    "llm_query",
    "llm_query_batched",
    "__dspy_construct__",
    "__dspy_call__",
    "__dspy_inputs__",
}
DEFAULT_LIMITS = {"max_duration_secs": 30.0, "max_memory": 64 * 1024 * 1024, "max_recursion_depth": 200}
CODE_KINDS = {"RLM", "CodeAct", "ProgramOfThought"}
RLM_DSPY_INSTRUCTIONS = """

This interpreter also provides a capability-limited `dspy` object. You may compose
real DSPy modules in your code, for example:
`child = dspy.RLM("context, question -> answer", max_iters=1)` followed by
`result = child(context=context, question=question)`. Available constructors are
Predict, ChainOfThought, RLM, CodeAct, ProgramOfThought, ReAct, and ReActV2.
Predictor results expose fields as attributes, such as `result.answer`.
"""

_ANNOTATION_NAMES = {
    "Any",
    "Literal",
    "NoneType",
    "Optional",
    "Union",
    "bool",
    "dict",
    "float",
    "int",
    "list",
    "set",
    "str",
    "tuple",
}


def _validate_signature_annotations(signature: str) -> None:
    """Reject guest annotations that could resolve imports in DSPy's parser."""
    for fields in signature.split("->"):
        function = ast.parse(f"def __dspy_signature({fields}): pass").body[0]
        if function.args.defaults or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
            raise ValueError("predictor signatures support only named fields without defaults")
        for argument in function.args.args:
            if argument.annotation is None:
                continue
            for node in ast.walk(argument.annotation):
                if isinstance(node, ast.Name) and node.id not in _ANNOTATION_NAMES:
                    raise ValueError(f"unsupported predictor annotation: {node.id}")
                if isinstance(node, (ast.Attribute, ast.Call)):
                    raise ValueError("predictor annotations cannot reference host modules or call functions")


def resolve_signature(value):
    if isinstance(value, str):
        _validate_signature_annotations(value)
        return value
    expected = {"__dspy_signature__", "signature", "instructions"}
    if not isinstance(value, dict) or set(value) != expected or value["__dspy_signature__"] is not True:
        raise ValueError("predictor signature must be a string or dspy.Signature marker")
    if not isinstance(value["signature"], str) or not isinstance(value["instructions"], (str, type(None))):
        raise ValueError("invalid dspy.Signature marker")
    _validate_signature_annotations(value["signature"])
    from dspy.signatures.signature import make_signature

    return make_signature(value["signature"], value["instructions"])


def jsonable(value: Any) -> Any:
    if isinstance(value, Example):
        value = value.toDict()
    return to_jsonable_python(value)


@dataclass(frozen=True)
class Policy:
    max_predictor_calls: int
    max_tool_calls: int
    max_nested_depth: int
    limits: dict[str, Any]
    request_timeout: float | None


class Budget:
    def __init__(self, policy: Policy):
        self.policy, self.predictors, self.tools, self.lock = policy, 0, 0, threading.Lock()

    def count(self, kind: str) -> None:
        with self.lock:
            attr, maximum = (
                ("predictors", self.policy.max_predictor_calls)
                if kind == "predictor"
                else ("tools", self.policy.max_tool_calls)
            )
            setattr(self, attr, getattr(self, attr) + 1)
            if getattr(self, attr) > maximum:
                raise RuntimeError(f"Monty program exceeded its limit of {maximum} {kind} calls")


def normalize_tools(values: list[Any]) -> dict[str, Tool]:
    result = {}
    for value in values:
        tool = value if isinstance(value, Tool) else Tool(value)
        if inspect.iscoroutinefunction(tool.func):
            raise ValueError("MontyProgram does not support asynchronous tools")
        if not tool.name.isidentifier() or keyword.iskeyword(tool.name):
            raise ValueError(f"invalid Monty tool name: {tool.name!r}")
        if tool.name in RESERVED or tool.name.startswith("__dspy"):
            raise ValueError(f"Monty tool name {tool.name!r} is reserved")
        if tool.name in result:
            raise ValueError(f"Duplicate Monty tool name: {tool.name!r}")
        result[tool.name] = tool
    return result


class Invocation:
    def __init__(
        self, *, policy: Policy, tools: dict[str, Tool], lm: Any, depth: int = 0, budget: Budget | None = None
    ):
        self.policy, self.tools, self.lm, self.depth = policy, tools, lm, depth
        self.budget = budget or Budget(policy)
        self.registry = {}

    def _int(self, config, name, default, maximum):
        value = config.pop(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 1 and {maximum}")
        return value

    def _tools(self, config, signature):
        values = config.pop("tools", [])
        if not isinstance(values, list):
            raise ValueError("tools must be a list of approved tools")
        selected = {}
        for value in values:
            # Monty represents an external callable by its lookup name when it
            # crosses back into a host callback.
            name = value if isinstance(value, str) else None
            if name not in self.tools:
                raise ValueError("tools may only reference approved host tools")
            selected[name] = self.tools[name]
        from dspy.signatures.signature import ensure_signature

        collisions = set(ensure_signature(signature).input_fields) & set(selected)
        if collisions:
            raise ValueError(f"predictor inputs conflict with selected tools: {sorted(collisions)}")
        return selected

    def _budgeted_tools(self, selected):
        result = []
        for tool in selected.values():

            def invoke(_tool=tool, **kwargs):
                self.budget.count("tool")
                return _tool(**kwargs)

            result.append(
                Tool(
                    invoke,
                    name=tool.name,
                    desc=tool.desc,
                    args=tool.args,
                    arg_types=tool.arg_types,
                    arg_desc=tool.arg_desc,
                )
            )
        return result

    def construct(self, kind, signature, config):
        if kind not in {"Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2"}:
            raise ValueError(f"unsupported predictor kind: {kind}")
        resolved_signature = resolve_signature(signature)
        if isinstance(resolved_signature, str) and resolved_signature.count("->") != 1:
            raise ValueError("predictor signature string must contain one '->'")
        if not isinstance(config, dict):
            raise ValueError("predictor configuration must be a dictionary")
        config = dict(config)
        from dspy import RLM, ChainOfThought, CodeAct, Predict, ProgramOfThought, ReAct, ReActV2

        types = {
            "Predict": Predict,
            "ChainOfThought": ChainOfThought,
            "RLM": RLM,
            "CodeAct": CodeAct,
            "ProgramOfThought": ProgramOfThought,
            "ReAct": ReAct,
            "ReActV2": ReActV2,
        }
        kwargs, selected_tools = {}, {}
        if kind in {"ReAct", "ReActV2", "RLM", "CodeAct"}:
            selected_tools = self._tools(config, resolved_signature)
            kwargs["tools"] = self._budgeted_tools(selected_tools)
        if kind in {"ReAct", "ReActV2", "RLM", "CodeAct", "ProgramOfThought"}:
            kwargs["max_iters"] = self._int(config, "max_iters", 20 if kind in {"ReAct", "ReActV2", "RLM"} else 5, 100)
        if kind == "RLM":
            kwargs["max_llm_calls"] = self._int(config, "max_llm_calls", 50, 100)
            kwargs["max_output_chars"] = self._int(config, "max_output_chars", 10_000, 100_000)
            verbose = config.pop("verbose", False)
            if not isinstance(verbose, bool):
                raise ValueError("verbose must be boolean")
            kwargs.update(verbose=verbose, sub_lm=self.lm)
        if config:
            raise ValueError(f"unsupported {kind} configuration: {sorted(config)}")
        if kind in CODE_KINDS:
            if self.depth >= self.policy.max_nested_depth:
                raise RuntimeError(f"code execution nesting exceeds maximum depth of {self.policy.max_nested_depth}")
            kwargs["interpreter_factory"] = self.interpreter_factory(selected_tools)
        predictor = types[kind](resolved_signature, **kwargs)
        if kind == "RLM":
            instructions = predictor.generate_action.signature.instructions + RLM_DSPY_INSTRUCTIONS
            predictor.generate_action.signature = predictor.generate_action.signature.with_instructions(instructions)
        if self.lm is not None:
            predictor.set_lm(self.lm)
        handle = f"predictor_{len(self.registry)}"
        self.registry[handle] = (kind, predictor)
        return handle

    def interpreter_factory(self, tools):
        def factory():
            from ._interpreter import MontyInterpreter

            return MontyInterpreter(self.child(tools))

        return factory

    def child(self, tools):
        return Invocation(policy=self.policy, tools=tools, lm=self.lm, depth=self.depth + 1, budget=self.budget)

    def call(self, handle, inputs):
        if set(inputs) & {"config", "demos", "lm", "new_signature", "signature", "sub_lm", "interpreter_factory"}:
            raise ValueError("predictor call cannot override control arguments")
        try:
            kind, predictor = self.registry[handle]
        except KeyError as error:
            raise ValueError("unknown or expired predictor handle") from error
        if kind in {"ReAct", "ReActV2", "CodeAct"} and "max_iters" in inputs:
            raise ValueError(f"{kind} max_iters can only be configured at construction")
        self.budget.count("predictor")
        return jsonable(predictor(**inputs))

    def lookup(self):
        result = {"__dspy_construct__": self.construct, "__dspy_call__": self.call}
        for name, tool in self.tools.items():

            def invoke(*args, _tool=tool, **kwargs):
                self.budget.count("tool")
                arguments = inspect.signature(_tool.func).bind(*args, **kwargs).arguments
                return jsonable(_tool(**arguments))

            result[name] = invoke
        return result
