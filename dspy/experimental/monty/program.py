"""Public Flex-style source program runtime."""

from __future__ import annotations

import pydantic

from dspy.adapters.utils import parse_value
from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental

from ._bridge import DEFAULT_LIMITS, Invocation, Policy, jsonable, normalize_tools
from ._shim import SHIM
from ._source import compile_source


@experimental
class MontyProgram(Module, Parameter):
    """Execute optimizable DSPy orchestration source in a Monty sandbox.

    ``module_src`` must define one ``dspy.Module`` subclass with ``__init__``
    and synchronous ``forward`` methods. The guest can compose ``Predict``,
    ``ChainOfThought``, ``RLM``, ``CodeAct``, ``ProgramOfThought``, ``ReAct``,
    and ``ReActV2`` using normal DSPy constructor and call syntax. Real modules,
    LMs, credentials, and tools remain in the host process; Monty receives only
    JSON-compatible values and opaque predictor/tool capabilities.

    A fresh invocation owns each outer sandbox and predictor registry. Nested
    code-executing modules receive distinct Monty sessions, selected tools only,
    and the invocation's shared predictor/tool budgets. Those budgets count
    bridged module and host-tool calls, not every internal LM turn in a compound
    module; ``max_iters`` and RLM's ``max_llm_calls`` bound those turns.

    If source is omitted, the baseline uses ``Predict`` or, when tools are
    supplied, ``RLM``. The source is this module's single opaque ``Parameter``.
    Async execution and custom signature annotations inside guest-authored
    predictors are not currently supported.
    """

    def __init__(
        self,
        signature,
        module_src=None,
        *,
        tools=None,
        max_predictor_calls=100,
        max_tool_calls=100,
        max_nested_depth=2,
        limits=None,
        request_timeout=120.0,
        callbacks=None,
    ):
        super().__init__(callbacks=callbacks)
        self._signature = ensure_signature(signature)
        self._tools = normalize_tools(tools or [])
        collisions = set(self.signature.input_fields) & set(self._tools)
        if collisions:
            raise ValueError(f"program inputs conflict with tools: {sorted(collisions)}")
        if module_src is None:
            module_src = self._baseline(bool(tools))
        self._module_src = module_src
        self._class_name, self._execution_src = compile_source(module_src)
        if not isinstance(max_predictor_calls, int) or max_predictor_calls < 1:
            raise ValueError("max_predictor_calls must be positive")
        if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        if not isinstance(max_nested_depth, int) or not 0 <= max_nested_depth <= 2:
            raise ValueError("max_nested_depth must be between 0 and 2")
        self._policy = Policy(
            max_predictor_calls, max_tool_calls, max_nested_depth, {**DEFAULT_LIMITS, **(limits or {})}, request_timeout
        )
        self._lm = None

    @property
    def signature(self):
        return self._signature

    @property
    def module_src(self):
        return self._module_src

    def _baseline(self, tools):
        inputs = ", ".join(self._signature.input_fields)
        kwargs = ", ".join(f"{name}={name}" for name in self._signature.input_fields)
        outputs = ", ".join(f"{name}=result.{name}" for name in self._signature.output_fields)
        kind = "RLM" if tools else "Predict"
        tool_config = f", tools=[{', '.join(self._tools)}]" if tools else ""
        predictor_signature = self._render_signature()
        signature_arg = f"dspy.Signature({predictor_signature!r}, {self._signature.instructions!r})"
        forward_args = f", {inputs}" if inputs else ""
        return (
            f"class MontyModule(dspy.Module):\n    def __init__(self):\n"
            f"        super().__init__()\n        self.p = dspy.{kind}({signature_arg}{tool_config})\n\n"
            f"    def forward(self{forward_args}):\n        result = self.p({kwargs})\n"
            f"        return dspy.Prediction({outputs})\n"
        )

    def _render_signature(self):
        from dspy.adapters.utils import get_annotation_name

        def render(fields):
            values = []
            for name, field in fields.items():
                annotation = get_annotation_name(field.annotation)
                try:
                    ensure_signature(f"value: {annotation} -> output")
                except Exception:
                    values.append(name)
                else:
                    values.append(f"{name}: {annotation}")
            return ", ".join(values)

        return f"{render(self.signature.input_fields)} -> {render(self.signature.output_fields)}"

    def _bind_code(self, source):
        name, execution = compile_source(source)
        self._module_src, self._class_name, self._execution_src = source, name, execution

    def named_predictors(self):
        return []

    def set_lm(self, lm):
        self._lm = lm

    def get_lm(self):
        return self._lm

    def reset(self):
        self._lm = None

    def dump_state(self, json_mode=True):
        return {"module_src": self.module_src, "lm": self._lm.dump_state() if self._lm is not None else None}

    def load_state(self, state, *, allow_unsafe_lm_state=False):
        self._bind_code(state["module_src"])
        lm_state = state.get("lm")
        if lm_state is None:
            self._lm = None
        else:
            from dspy.clients.base_lm import BaseLM
            from dspy.predict.predict import _sanitize_lm_state

            safe = _sanitize_lm_state(lm_state, allow_unsafe_lm_state)
            self._lm = BaseLM.load_state(safe, allow_custom_lm_class=allow_unsafe_lm_state)
        return self

    def forward(self, **input_args):
        missing = set(self.signature.input_fields) - set(input_args)
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")
        try:
            from pydantic_monty import Monty
        except ImportError as error:
            raise ImportError("MontyProgram requires `pip install 'dspy[monty]'`") from error
        invocation = Invocation(policy=self._policy, tools=self._tools, lm=self._lm)
        source = (
            SHIM + "\n" + self._execution_src + f"\n__dspy_module = {self._class_name}()\n"
            "_dspy_fields(__dspy_module.forward(**__dspy_inputs__))\n"
        )
        inputs = {name: input_args[name] for name in self.signature.input_fields}
        with Monty(request_timeout=self._policy.request_timeout) as pool:
            with pool.checkout(limits=self._policy.limits) as session:
                output = session.feed_run(
                    source,
                    inputs={"__dspy_inputs__": jsonable(inputs)},
                    external_lookup=invocation.lookup(),
                )
        if not isinstance(output, dict):
            raise TypeError("Monty forward() must return dspy.Prediction or dict")
        missing = set(self.signature.output_fields) - set(output)
        if missing:
            raise ValueError(f"Monty forward() did not return output fields: {sorted(missing)}")
        result, errors = {}, []
        for name, field in self.signature.output_fields.items():
            try:
                result[name] = parse_value(output[name], field.annotation)
            except (TypeError, ValueError, pydantic.ValidationError) as error:
                errors.append(f"{name}: {error}")
        if errors:
            raise ValueError("Monty returned invalid output fields: " + "; ".join(errors))
        return Prediction(**result)

    async def aforward(self, **input_args):
        raise NotImplementedError("MontyProgram supports synchronous forward only")
